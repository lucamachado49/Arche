"""
scripts/diag_riskfactor.py — a curvatura serve como fator de RISCO de co-queda?

O guardrail (diag_fragility) mostrou que o NÍVEL de κ prevê a participação do
ativo nas quedas do mercado ALÉM da volatilidade (t=5,3). Aqui testamos se isso
é PRÁTICO:

  Teste A — double-sort: dentro de cada faixa de volatilidade, κ separa os que
            caem mais/menos na queda do mercado? (prova que é incremental e usável)
  Teste B — overlay defensivo: um tilt long-only que favorece κ "defensivo"
            protege mais na queda do que um tilt de baixa-vol? E o κ ortogonal
            à vol agrega sobre a vol?

Sinal (dos dados): κ ALTO → cai MENOS na queda do mercado → defensivo.
Look-ahead safe: κ e vol em t (point-in-time); retornos de t+1 em diante.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()

from src.data.config import PROJECT_ROOT
from src.data.cleaner import DataCleaner

H = 21


def month_ends(idx):
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def fwd_downpart(returns, mkt, dates, h=H):
    """Retorno médio do ativo nos dias em que o mercado caiu, t+1..t+h."""
    idx = returns.index
    out = {}
    for d in dates:
        pos = idx.searchsorted(d, side="right")
        seg = returns.iloc[pos:pos + h]
        mseg = mkt.iloc[pos:pos + h]
        if len(seg) < h // 2 or (mseg < 0).sum() < 3:
            continue
        out[d] = seg[(mseg < 0).values].mean()
    return pd.DataFrame(out).T


def zc(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def perf(r, mkt):
    r = r.dropna(); m = mkt.reindex(r.index)
    cum = r.cumsum(); dd = float((cum - cum.cummax()).min())
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    down = m < 0
    dc = r[down].mean() / m[down].mean() if m[down].mean() != 0 else np.nan
    return dict(ret=r.mean() * 252, vol=r.std() * np.sqrt(252), sharpe=sh, maxdd=dd, dcap=dc)


def main():
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
    prices.index = pd.to_datetime(prices.index); prices = prices.sort_index()
    kappa = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "transfer" / "BR" / "kappa_ts.parquet")
    kappa.index = pd.to_datetime(kappa.index); kappa = kappa.sort_index()
    common = [c for c in kappa.columns if c in prices.columns]
    kappa = kappa[common]
    returns = DataCleaner().prices_to_log_returns(prices[common])
    mkt = returns.mean(axis=1)

    rebal = month_ends(kappa.index)
    kap = kappa.loc[kappa.index.isin(rebal)]
    tvol = pd.DataFrame({d: returns.loc[returns.index <= d].tail(60).std() for d in kap.index}).T
    dp = fwd_downpart(returns, mkt, kap.index)

    # ---------- TESTE A: double-sort κ dentro de faixas de vol ----------
    print("\n" + "═" * 66)
    print("TESTE A — participação na queda por tercil de κ, dentro de faixas de vol")
    print("(valor = retorno médio em dias de mercado ruim; MENOS negativo = mais defensivo)")
    print("═" * 66)
    rows = []
    for d in kap.index.intersection(dp.index):
        k, v, y = kap.loc[d], tvol.loc[d], dp.loc[d]
        m = k.notna() & v.notna() & y.notna()
        if m.sum() < 20:
            continue
        vt = pd.qcut(v[m].rank(method="first"), 3, labels=["vol baixa", "vol média", "vol alta"])
        kt = pd.qcut(k[m].rank(method="first"), 3, labels=["κ baixo", "κ médio", "κ alto"])
        for a in k[m].index:
            rows.append((vt[a], kt[a], y[a]))
    df = pd.DataFrame(rows, columns=["vol", "kappa", "downpart"])
    tbl = df.groupby(["vol", "kappa"], observed=True)["downpart"].mean().unstack()
    tbl = tbl.reindex(index=["vol baixa", "vol média", "vol alta"],
                      columns=["κ baixo", "κ médio", "κ alto"])
    print(f"{'':<12}{'κ baixo':>12}{'κ médio':>12}{'κ alto':>12}{'  Δ(alto-baixo)':>16}")
    for vlab in tbl.index:
        r = tbl.loc[vlab]
        d_ka = r['κ alto'] - r['κ baixo']
        print(f"{vlab:<12}{r['κ baixo']:>12.5f}{r['κ médio']:>12.5f}{r['κ alto']:>12.5f}{d_ka:>+16.5f}")
    print("\nΔ positivo em cada linha = κ alto cai MENOS que κ baixo NA MESMA faixa de vol")
    print("=> κ separa risco de co-queda ALÉM da volatilidade.")

    # ---------- TESTE B: overlays defensivos head-to-head ----------
    LAM = 0.9
    def build(signal_fn, label):
        w = {}
        for d in kap.index:
            k, v = kap.loc[d].dropna(), tvol.loc[d]
            common_i = k.index.intersection(v.dropna().index)
            if len(common_i) < 15:
                continue
            s = signal_fn(k.loc[common_i], v.loc[common_i])
            wb = pd.Series(1.0 / len(common_i), index=common_i)
            wi = wb * np.exp(LAM * zc(s))
            w[d] = wi / wi.sum()
        W = pd.DataFrame(w).T.fillna(0.0)
        cc = [c for c in W.columns if c in returns.columns]
        Wd = W[cc].reindex(returns.index).ffill()
        r = (returns[cc] * Wd.shift(1)).sum(axis=1)
        return r.loc[r.index >= W.index[0]], label

    def resid_k(k, v):  # κ ortogonal à vol (cross-seccional)
        x = zc(v).values; y = zc(k).values
        b = np.dot(x, y) / np.dot(x, x) if np.dot(x, x) > 0 else 0
        return pd.Series(y - b * x, index=k.index)

    overlays = [
        (lambda k, v: pd.Series(0.0, index=k.index), "Base (equal-weight)"),
        (lambda k, v: -zc(v), "Defensivo VOL (baixa-vol)"),
        (lambda k, v: zc(k), "Defensivo κ (curvatura alta)"),
        (lambda k, v: resid_k(k, v), "Defensivo κ⊥vol (incremental)"),
        (lambda k, v: zc(-zc(v) + zc(k)), "Combinado VOL + κ"),
    ]
    print("\n" + "═" * 66)
    print(f"TESTE B — overlay defensivo long-only (tilt exp, λ={LAM}) vs mercado EW")
    print("dcap = downside capture (quanto cai vs mercado na queda; MENOR = protege mais)")
    print("═" * 66)
    print(f"{'overlay':<32}{'ret_aa':>8}{'vol':>7}{'sharpe':>8}{'maxDD':>9}{'dcap':>7}")
    base_r = None
    for fn, lab in overlays:
        r, _ = build(fn, lab)
        if base_r is None:
            base_r = r
        p = perf(r, mkt)
        print(f"{lab:<32}{p['ret']:>8.2%}{p['vol']:>7.1%}{p['sharpe']:>8.2f}{p['maxdd']:>9.1%}{p['dcap']:>7.2f}")


if __name__ == "__main__":
    main()
