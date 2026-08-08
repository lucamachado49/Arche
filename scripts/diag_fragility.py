"""
scripts/diag_fragility.py — e se a curvatura mede RISCO, não retorno?

Reframe: o sinal de retorno (GDS) é fraco e concentrado em 2020. A literatura
(Sandhu 2016) diz que a curvatura de Ricci mede FRAGILIDADE sistêmica —
curvatura alta = ativos hipersincronizados que caem juntos na crise.

Teste: o NÍVEL da curvatura nodal κᵢ prevê o RISCO DE QUEDA futuro do ativo,
cross-seccionalmente? E — o teste decisivo — isso sobrevive a EXCLUIR 2020
(ao contrário do sinal de retorno)?

Alvos futuros (t+1..t+h, h=21), por ativo:
  semidev  : desvio-padrão só dos retornos negativos (downside vol)
  worst    : pior retorno diário da janela
  downpart : retorno médio do ativo nos dias em que o MERCADO caiu (participação na queda)

Sinais: kappa_level = κᵢ(t)  |  kappa_change = Δκᵢ (o GDS antigo)
IC = Spearman(sinal_t, alvo futuro). Reporta IC médio, t-stat e IC ex-2020.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

logger.remove()

from src.data.config import PROJECT_ROOT
from src.data.cleaner import DataCleaner

H = 21


def fwd_risk(returns: pd.DataFrame, mkt: pd.Series, dates, h: int = H):
    """Métricas de risco de queda t+1..t+h por ativo, para cada data semanal."""
    idx = returns.index
    semidev, worst, downpart = {}, {}, {}
    for d in dates:
        pos = idx.searchsorted(d, side="right")
        seg = returns.iloc[pos:pos + h]
        mseg = mkt.iloc[pos:pos + h]
        if len(seg) < h // 2:
            continue
        neg = seg.where(seg < 0)
        semidev[d] = neg.std()
        worst[d] = seg.min()
        down_days = mseg < 0
        downpart[d] = seg[down_days.values].mean() if down_days.sum() >= 3 else seg.mean() * np.nan
    return (pd.DataFrame(semidev).T, pd.DataFrame(worst).T, pd.DataFrame(downpart).T)


def ic_summary(sig: pd.DataFrame, tgt: pd.DataFrame, minn: int = 10):
    ics = {}
    for d in sig.index.intersection(tgt.index):
        s, t = sig.loc[d], tgt.loc[d]
        m = s.notna() & t.notna()
        if m.sum() < minn:
            continue
        rho, _ = stats.spearmanr(s[m], t[m])
        if np.isfinite(rho):
            ics[d] = rho
    ics = pd.Series(ics).dropna()
    def st(x):
        x = x.dropna()
        return (x.mean(), x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 10 else (np.nan, np.nan)
    m_all, t_all = st(ics)
    m_ex, t_ex = st(ics[ics.index.year != 2020])
    return m_all, t_all, m_ex, t_ex


def main():
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
    prices.index = pd.to_datetime(prices.index); prices = prices.sort_index()
    kappa = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "transfer" / "BR" / "kappa_ts.parquet")
    kappa.index = pd.to_datetime(kappa.index); kappa = kappa.sort_index()
    common = [c for c in kappa.columns if c in prices.columns]
    kappa = kappa[common]
    returns = DataCleaner().prices_to_log_returns(prices[common])
    mkt = returns.mean(axis=1)

    semidev, worst, downpart = fwd_risk(returns, mkt, kappa.index)

    signals = {"kappa_level": kappa, "kappa_change (GDS)": kappa.diff(1)}
    targets = {"semidev (+ = mais risco)": semidev,
               "worst (- = mais risco)": worst,
               "downpart (- = cai mais na queda)": downpart}

    print(f"\nIC cross-seccional κ × RISCO futuro (h={H}d) | {len(kappa)} semanas")
    print("═" * 84)
    print(f"{'sinal':<20}{'alvo de risco':<32}{'IC':>9}{'t':>7}   |  {'IC ex-2020':>11}{'t':>7}")
    print("-" * 84)
    for sname, sig in signals.items():
        for tname, tgt in targets.items():
            m, t, mx, tx = ic_summary(sig, tgt)
            print(f"{sname:<20}{tname:<32}{m:>+9.4f}{t:>7.2f}   |  {mx:>+11.4f}{tx:>7.2f}")
        print("-" * 84)

    # ── GUARDRAIL: κ prevê risco ALÉM da volatilidade? (Fama-MacBeth) ──
    # vol trailing 60d point-in-time por ativo
    tvol = {d: returns.loc[returns.index <= d].tail(60).std() for d in kappa.index}
    tvol = pd.DataFrame(tvol).T

    def fama_macbeth(tgt: pd.DataFrame):
        """Coef médio (e t) de κ e da vol na regressão cross-seccional semanal do alvo."""
        ck, cv = [], []
        for d in kappa.index.intersection(tgt.index):
            k, v, y = kappa.loc[d], tvol.loc[d], tgt.loc[d]
            m = k.notna() & v.notna() & y.notna()
            if m.sum() < 15:
                continue
            def z(x):
                s = x[m]; sd = s.std()
                return (s - s.mean()) / sd if sd > 1e-12 else s * 0
            X = np.column_stack([np.ones(m.sum()), z(v).values, z(k).values])
            beta, *_ = np.linalg.lstsq(X, y[m].values, rcond=None)
            cv.append(beta[1]); ck.append(beta[2])
        cv, ck = np.array(cv), np.array(ck)
        def stt(a): return (a.mean(), a.mean() / (a.std() / np.sqrt(len(a))))
        return stt(cv), stt(ck), len(ck)

    print("\nGUARDRAIL — Fama-MacBeth: alvo ~ vol_trailing + κ (padronizados)")
    print("κ sobrevive ao controle por volatilidade?")
    print("═" * 84)
    print(f"{'alvo de risco':<32}{'coef vol':>10}{'t vol':>8}   {'coef κ':>10}{'t κ':>8}{'n sem':>8}")
    print("-" * 84)
    for tname, tgt in targets.items():
        (mv, tv), (mk, tk), n = fama_macbeth(tgt)
        print(f"{tname:<32}{mv:>+10.4f}{tv:>8.1f}   {mk:>+10.4f}{tk:>8.1f}{n:>8}")


if __name__ == "__main__":
    main()
