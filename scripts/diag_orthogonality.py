"""
scripts/diag_orthogonality.py — o alpha do GDS é genuíno ou fator conhecido?

Regride os retornos BRUTOS do L/S do GDS contra fatores construídos do próprio
painel de preços (sem depender do NEFIN):
  MKT  : retorno EW do universo (mercado)
  STR  : short-term reversal  (long perdedores 21d, short vencedores 21d)
  MOM  : momentum 12-1m       (long vencedores, short perdedores)
  LOWV : low-vol              (long baixa vol 60d, short alta vol)

Suspeitos nº1 de um sinal de VARIAÇÃO de curvatura: STR e LOWV. Se o alpha
do GDS sobrevive (t_NW > 2) a esses controles, há informação incremental —
e vale o teste oficial FF4/NEFIN. Se morre, é fator reembalado.

t-stat com erros Newey-West (HAC). Look-ahead safe: características em t usam
só retornos ≤ t; pesos de t rendem em t+1.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

logger.remove()

from src.data.config import PROJECT_ROOT
from src.data.cleaner import DataCleaner
from src.signal.generator import SignalGenerator
from src.backtest.walkforward import WalkForwardBacktest


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def ls_factor(char: pd.DataFrame, returns: pd.DataFrame, rebal: pd.DatetimeIndex,
              q: float = 0.2) -> pd.Series:
    """Fator L/S EW 'alto menos baixo' da característica, rebal mensal, hold diário."""
    weights = {}
    for d in rebal:
        c = char.loc[d].dropna() if d in char.index else None
        if c is None or len(c) < 10:
            continue
        hi = c[c >= c.quantile(1 - q)].index
        lo = c[c <= c.quantile(q)].index
        if len(hi) == 0 or len(lo) == 0:
            continue
        w = pd.Series(0.0, index=c.index)
        w[hi] = 1.0 / len(hi)
        w[lo] = -1.0 / len(lo)
        weights[d] = w
    W = pd.DataFrame(weights).T.reindex(returns.index).ffill()
    common = [c for c in W.columns if c in returns.columns]
    return (returns[common] * W[common].shift(1)).sum(axis=1)


def main() -> None:
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    cache_dir = PROJECT_ROOT / "data" / "cache" / "transfer" / "BR"
    _, gds = SignalGenerator(cache_dir=cache_dir).run(prices=prices, use_cache=True)
    common = [c for c in gds.columns if c in prices.columns]
    prices, gds = prices[common], gds[common]
    returns = DataCleaner().prices_to_log_returns(prices)

    # ── y: gross do L/S GDS (engine contínuo, config oficial) ──
    # Amostra INTEIRA (2013–24) p/ máximo poder estatístico na regressão
    # descritiva — sem trade aqui, só medição de ortogonalidade.
    eng = WalkForwardBacktest()
    eng.oos_start = gds.index[0]   # amostra inteira; warmup da EWMA é interno
    y = eng.run_continuous(returns=returns, gds_ts=gds, prices=prices)["gross"]

    # ── características point-in-time (rolling, causais) ──
    rev21 = returns.rolling(21).sum()                       # retorno 21d
    mom   = returns.rolling(252).sum() - returns.rolling(21).sum()  # 12-1m
    vol60 = returns.rolling(60).std()
    rebal = month_ends(returns.loc[returns.index >= y.index[0]].index)

    F = pd.DataFrame({
        "MKT":  returns[common].reindex(y.index).mean(axis=1),
        "STR":  -ls_factor(rev21, returns, rebal),   # long perdedores
        "MOM":  ls_factor(mom, returns, rebal),
        "LOWV": -ls_factor(vol60, returns, rebal),   # long baixa vol
    }).reindex(y.index).dropna()

    df = pd.concat([y.rename("GDS"), F], axis=1).dropna()
    X = sm.add_constant(df[["MKT", "STR", "MOM", "LOWV"]])
    ols = sm.OLS(df["GDS"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 21})

    ann = 252
    print("\n" + "═" * 66)
    print("ORTOGONALIDADE — gross L/S GDS ~ MKT + STR + MOM + LOWV (HAC)")
    print("═" * 66)
    print(f"{'termo':<8}{'beta':>12}{'t_NW':>10}{'p':>10}")
    for name in X.columns:
        b, t, p = ols.params[name], ols.tvalues[name], ols.pvalues[name]
        label = "ALPHA" if name == "const" else name
        extra = f"   (α a.a. = {b*ann:+.2%})" if name == "const" else ""
        print(f"{label:<8}{b:>12.6f}{t:>10.2f}{p:>10.4f}{extra}")
    print("-" * 66)
    print(f"R² = {ols.rsquared:.3f} | n = {int(ols.nobs)} dias")

    # Correlação simples do GDS com cada fator (diagnóstico)
    print("\nCorrelação diária GDS×fator:")
    for f in ["MKT", "STR", "MOM", "LOWV"]:
        print(f"  {f:<5}: {df['GDS'].corr(df[f]):+.3f}")


if __name__ == "__main__":
    main()
