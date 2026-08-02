"""
scripts/diag_stress.py — Ricci Risk / Geometric Stress Monitor.

Pivô: de "GDS = alpha cross-seccional" para "geometria AGREGADA do mercado =
detector de estado frágil". Testa se medidas agregadas da rede de correlações
PREVEEM vol e drawdown futuros, e se um overlay de risco (reduzir exposição
quando a geometria sinaliza estresse) melhora uma estratégia tradicional.

Medidas agregadas (por data semanal, point-in-time):
  kbar      : curvatura nodal média do mercado (Sandhu: sobe com risco sistêmico)
  kdisp     : dispersão cross-seccional da curvatura nodal
  gds_absm  : |GDS| médio (intensidade da variação de curvatura)
  n_extreme : nº de ativos com |GDS| > 2 (quantos nós estão deslocando)

Alvos futuros (mercado = EW do universo), janela h=21 dias úteis:
  fwd_vol   : vol realizada t+1..t+h (anualizada)
  fwd_ret   : retorno t+1..t+h
  fwd_maxdd : pior drawdown intra-janela t+1..t+h

Tudo causal: medida em t usa dados ≤ t; alvo é estritamente futuro.
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


def fwd_metrics(mkt: pd.Series, dates: pd.DatetimeIndex, h: int = H) -> pd.DataFrame:
    """Vol, retorno e pior drawdown da janela (t, t+h] para cada data."""
    idx = mkt.index
    rows = {}
    for d in dates:
        pos = idx.searchsorted(d, side="right")
        seg = mkt.iloc[pos:pos + h]
        if len(seg) < h // 2:
            continue
        cum = seg.cumsum()
        peak = cum.cummax()
        rows[d] = {
            "fwd_vol": seg.std() * np.sqrt(252),
            "fwd_ret": seg.sum(),
            "fwd_maxdd": float((cum - peak).min()),
        }
    return pd.DataFrame(rows).T


def expanding_z(x: pd.Series, minp: int = 52) -> pd.Series:
    mu = x.expanding(min_periods=minp).mean().shift(1)
    sd = x.expanding(min_periods=minp).std().shift(1)
    return (x - mu) / sd


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    cum = r.cumsum()
    dd = float((cum - cum.cummax()).min())
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    return dict(ret=r.mean() * 252, vol=r.std() * np.sqrt(252), sharpe=sh, maxdd=dd)


def main():
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
    prices.index = pd.to_datetime(prices.index); prices = prices.sort_index()
    kappa = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "transfer" / "BR" / "kappa_ts.parquet")
    gds = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "transfer" / "BR" / "gds_ts.parquet")
    kappa.index = pd.to_datetime(kappa.index); gds.index = pd.to_datetime(gds.index)

    common = [c for c in kappa.columns if c in prices.columns]
    returns = DataCleaner().prices_to_log_returns(prices[common])
    mkt = returns.mean(axis=1).dropna()                 # proxy de mercado (EW)

    # ── medidas agregadas (semanais) ──
    agg = pd.DataFrame({
        "kbar":      kappa[common].mean(axis=1),
        "kdisp":     kappa[common].std(axis=1),
        "gds_absm":  gds[common].abs().mean(axis=1),
        "n_extreme": (gds[common].abs() > 2).sum(axis=1),
    }).dropna()

    fwd = fwd_metrics(mkt, agg.index)
    df = agg.join(fwd, how="inner").dropna()

    print(f"\nObs: {len(df)} semanas | mercado EW | h={H}d")
    print("═" * 72)
    print("PODER PREDITIVO (Spearman: medida em t × alvo futuro t+1..t+h)")
    print("═" * 72)
    print(f"{'medida':<12}{'→ fwd_vol':>14}{'→ fwd_ret':>14}{'→ fwd_maxdd':>14}")
    for m in agg.columns:
        row = ""
        for tgt in ("fwd_vol", "fwd_ret", "fwd_maxdd"):
            rho, p = stats.spearmanr(df[m], df[tgt])
            star = "*" if p < 0.01 else (" " if p < 0.05 else " ")
            row += f"{rho:>+10.3f}{('('+f'{p:.3f}'+')'):>0}"
        print(f"{m:<12}" + "  ".join(
            f"{stats.spearmanr(df[m], df[t])[0]:>+7.3f} p={stats.spearmanr(df[m], df[t])[1]:.3f}"
            for t in ("fwd_vol", "fwd_ret", "fwd_maxdd")))

    # ── TESTE CRÍTICO: kbar prevê vol futura ALÉM da vol corrente? ──
    # Se kbar é só proxy da vol atual (autocorrelacionada), não há info nova.
    import statsmodels.api as sm
    tvol = mkt.rolling(21).std().reindex(df.index) * np.sqrt(252)   # vol corrente
    reg = pd.DataFrame({
        "fwd_vol": df["fwd_vol"],
        "vol_atual": tvol,
        "kbar": df["kbar"],
    }).dropna()
    zc = (reg[["vol_atual", "kbar"]] - reg[["vol_atual", "kbar"]].mean()) / reg[["vol_atual", "kbar"]].std()
    X = sm.add_constant(zc)
    ols = sm.OLS(reg["fwd_vol"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 21})
    print("\n" + "═" * 72)
    print("INCREMENTAL — fwd_vol ~ vol_atual + kbar (z-scores, HAC). kbar sobrevive?")
    print("═" * 72)
    for name in X.columns:
        print(f"  {name:<10} beta={ols.params[name]:>+.4f}  t_NW={ols.tvalues[name]:>+.2f}  p={ols.pvalues[name]:.4f}")
    print(f"  R²={ols.rsquared:.3f} | n={int(ols.nobs)}")

    # ── escolhe a medida que melhor prevê vol futura, orienta e roda overlay ──
    best = max(agg.columns, key=lambda m: abs(stats.spearmanr(df[m], df["fwd_vol"])[0]))
    sign = np.sign(stats.spearmanr(df[best], df["fwd_vol"])[0])
    print("\n" + "═" * 72)
    print(f"OVERLAY DE RISCO — sinal = {best} (orientado p/ estresse), exposição contínua")
    print("═" * 72)

    def run_overlay(stress_weekly_or_daily: pd.Series) -> tuple[pd.Series, pd.Series]:
        z = expanding_z(stress_weekly_or_daily).reindex(returns.index).ffill()
        expo = (1.0 - z / 2.0).clip(lower=0.0, upper=1.0)
        ov = (expo.shift(1) * mkt).dropna()
        return ov, expo

    # Sinal A: curvatura agregada (geometria)  |  Sinal B: vol realizada simples
    ov_k, expo_k = run_overlay(sign * agg[best])
    ov_v, expo_v = run_overlay(mkt.rolling(21).std())   # baseline: vol-targeting puro

    bh = mkt
    ci = bh.index.intersection(ov_k.index).intersection(ov_v.index)
    bh, ov_k, ov_v = bh.loc[ci], ov_k.loc[ci], ov_v.loc[ci]

    pb, pk, pv = perf(bh), perf(ov_k), perf(ov_v)
    print(f"{'':<22}{'ret_aa':>9}{'vol_aa':>9}{'sharpe':>9}{'maxDD':>9}")
    print(f"{'buy&hold':<22}{pb['ret']:>9.2%}{pb['vol']:>9.2%}{pb['sharpe']:>9.2f}{pb['maxdd']:>9.2%}")
    print(f"{'+ overlay CURVATURA':<22}{pk['ret']:>9.2%}{pk['vol']:>9.2%}{pk['sharpe']:>9.2f}{pk['maxdd']:>9.2%}")
    print(f"{'+ overlay VOL (base)':<22}{pv['ret']:>9.2%}{pv['vol']:>9.2%}{pv['sharpe']:>9.2f}{pv['maxdd']:>9.2%}")
    print("\nO teste que importa: curvatura BATE o vol-targeting simples?")
    print(f"  Δsharpe (curv − vol): {pk['sharpe']-pv['sharpe']:+.3f} | "
          f"ΔmaxDD (curv − vol): {pk['maxdd']-pv['maxdd']:+.1%}")
    print(f"  corr(exposição_curv, exposição_vol) = {expo_k.reindex(ci).corr(expo_v.reindex(ci)):+.2f}")


if __name__ == "__main__":
    main()
