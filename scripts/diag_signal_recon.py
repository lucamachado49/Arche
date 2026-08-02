"""
scripts/diag_signal_recon.py — reconstrução do sinal a partir de κ_i (cache).

O GDS atual = Δκ de 1 semana / σ_i, dominado por ruído de rewiring da MST.
Testa definições alternativas do sinal usando a curvatura nodal κ_i já em
cache (sem recomputar Ollivier-Ricci), comparando o IC cross-seccional a
h=21d — métrica do Sprint 4 — com o filtro honesto ex-2020.

Variantes (cada uma: DataFrame data×ativo):
  diff1   : κ.diff(1)  / σ_i        (≈ GDS atual)
  diff4   : κ.diff(4)  / σ_i        (baseline ~1 mês)
  diff8   : κ.diff(8)  / σ_i
  diff12  : κ.diff(12) / σ_i        (baseline ~1 trimestre)
  leveldev: (κ − EWMA12(κ)) / σ_i   (nível vs tendência lenta)
  levelz  : z-score cross-seccional de κ (nível cru)
  ewmdiff : Δ de κ suavizado (EWMA20) — o que o backtest usa hoje

σ_i = desvio-padrão expansível de κ_i (look-ahead safe). IC = Spearman(sinal_t,
retorno t+1..t+h). Reporta IC médio, t-stat e IC ex-2020.
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
MINH = 24   # min obs p/ σ expansível


def fwd_returns(returns: pd.DataFrame, dates: pd.DatetimeIndex, h: int = H) -> pd.DataFrame:
    """Retorno acumulado t+1..t+h por ativo, para cada data semanal."""
    idx = returns.index
    rows = {}
    for d in dates:
        pos = idx.searchsorted(d, side="right")
        seg = returns.iloc[pos:pos + h]
        if len(seg) < h // 2:
            continue
        rows[d] = seg.sum()
    return pd.DataFrame(rows).T


def ic_series(signal: pd.DataFrame, fwd: pd.DataFrame, minn: int = 10) -> pd.Series:
    out = {}
    for d in signal.index.intersection(fwd.index):
        s, f = signal.loc[d], fwd.loc[d]
        m = s.notna() & f.notna()
        if m.sum() < minn:
            continue
        rho, _ = stats.spearmanr(s[m], f[m])
        if np.isfinite(rho):
            out[d] = rho
    return pd.Series(out)


def summ(ic: pd.Series) -> tuple[float, float, int]:
    ic = ic.dropna()
    if len(ic) < 10:
        return (np.nan, np.nan, len(ic))
    return (ic.mean(), ic.mean() / (ic.std() / np.sqrt(len(ic))), len(ic))


def main():
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
    prices.index = pd.to_datetime(prices.index); prices = prices.sort_index()
    kappa = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "transfer" / "BR" / "kappa_ts.parquet")
    kappa.index = pd.to_datetime(kappa.index); kappa = kappa.sort_index()
    common = [c for c in kappa.columns if c in prices.columns]
    kappa = kappa[common]
    returns = DataCleaner().prices_to_log_returns(prices[common])

    sigma = kappa.expanding(min_periods=MINH).std()          # σ_i expansível
    def cs_z(df):                                            # z-score cross-seccional
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)

    variants = {
        "diff1":    kappa.diff(1) / sigma,
        "diff4":    kappa.diff(4) / sigma,
        "diff8":    kappa.diff(8) / sigma,
        "diff12":   kappa.diff(12) / sigma,
        "leveldev": (kappa - kappa.ewm(span=12, adjust=False).mean()) / sigma,
        "levelz":   cs_z(kappa),
        "ewmdiff":  (kappa.ewm(span=20, adjust=False).mean()).diff(1) / sigma,
    }

    fwd = fwd_returns(returns, kappa.index)

    print(f"\nIC cross-seccional h={H}d | {len(fwd)} datas | reconstrução do sinal")
    print("═" * 70)
    print(f"{'variante':<12}{'IC_méd':>9}{'t-stat':>9}{'n':>6}   |  {'IC ex-2020':>11}{'t':>7}")
    print("-" * 70)
    for name, sig in variants.items():
        ic = ic_series(sig, fwd)
        m, t, n = summ(ic)
        mx, tx, nx = summ(ic[ic.index.year != 2020])
        print(f"{name:<12}{m:>+9.4f}{t:>9.2f}{n:>6}   |  {mx:>+11.4f}{tx:>7.2f}")


if __name__ == "__main__":
    main()
