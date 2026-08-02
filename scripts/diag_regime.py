"""
scripts/diag_regime.py — o GDS é um alpha CONDICIONAL de regime?

Dois testes, ambos com a coluna "ex-2020" (o filtro honesto: se um regime só
funciona porque contém 2020, não é efeito de regime — é o evento COVID).

A) Calendário (a grade pedida: pré/pós-crise, pré/pós-Covid, inflação, juros).
B) Condicionamento POINT-IN-TIME pela geometria do mercado: tercis de
   (i) correlação média par-a-par e (ii) vol de mercado, medidos no INÍCIO de
   cada mês (dados ≤ mês anterior). Testa a tese "GDS paga quando a rede de
   correlações disloca" de forma não-circular e robusta a snooping.

Retorno do L/S GDS é log mensal; estatística t simples por bucket.
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
from src.signal.generator import SignalGenerator
from src.backtest.walkforward import WalkForwardBacktest


def tstat(r: pd.Series) -> tuple[float, float, int]:
    r = r.dropna()
    if len(r) < 6:
        return (np.nan, np.nan, len(r))
    return (r.mean() * 12, r.mean() / (r.std() / np.sqrt(len(r))), len(r))


def line(label: str, r_all: pd.Series, ex2020: pd.Series) -> None:
    a, ta, na = tstat(r_all)
    b, tb, nb = tstat(ex2020)
    print(f"{label:<26}{na:>4} {a:>8.2%} {ta:>6.2f}   |  {nb:>4} {b:>8.2%} {tb:>6.2f}")


def avg_pairwise_corr(win: pd.DataFrame) -> float:
    w = win.dropna(axis=1, thresh=int(0.8 * len(win)))
    if w.shape[1] < 5:
        return np.nan
    c = w.corr().values
    n = c.shape[0]
    return (np.nansum(c) - n) / (n * (n - 1))


def main() -> None:
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    cache_dir = PROJECT_ROOT / "data" / "cache" / "transfer" / "BR"
    _, gds = SignalGenerator(cache_dir=cache_dir).run(prices=prices, use_cache=True)
    common = [c for c in gds.columns if c in prices.columns]
    prices, gds = prices[common], gds[common]
    returns = DataCleaner().prices_to_log_returns(prices)

    eng = WalkForwardBacktest()
    eng.oos_start = gds.index[0]
    y = eng.run_continuous(returns=returns, gds_ts=gds, prices=prices)["gross"]

    m = y.resample("ME").sum()                       # retorno log mensal do L/S
    m = m[m != 0.0]
    ex = m[m.index.year != 2020]

    hdr = f"{'regime':<26}{'n':>4} {'ret_aa':>8} {'t':>6}   |  {'n':>4} {'ret_aa':>8} {'t':>6}"
    print("\n" + "═" * 74)
    print("A) REGIMES DE CALENDÁRIO          AMOSTRA CHEIA        |       EX-2020")
    print("═" * 74)
    print(hdr)
    print("-" * 74)
    idx = m.index
    regimes = {
        "pré-crise (<2014)":        idx.year < 2014,
        "crise BR (2014–2016)":     (idx.year >= 2014) & (idx.year <= 2016),
        "pós-crise (2017–2019)":    (idx.year >= 2017) & (idx.year <= 2019),
        "pré-Covid (<2020-02)":     idx < pd.Timestamp("2020-02-01"),
        "Covid (2020)":             idx.year == 2020,
        "pós-Covid (2021+)":        idx.year >= 2021,
        "inflação alta (15-16,21-22)": np.isin(idx.year, [2015, 2016, 2021, 2022]),
        "juros baixos (2020-2021)": np.isin(idx.year, [2020, 2021]),
    }
    for label, mask in regimes.items():
        seg = m[mask]
        line(label, seg, seg[seg.index.year != 2020])

    # ── B) Condicionamento point-in-time pela geometria ──
    cols = returns.columns
    corr_state, vol_state = {}, {}
    for d in m.index:
        win = returns.loc[returns.index < d.replace(day=1)].tail(60)
        if len(win) < 40:
            continue
        corr_state[d] = avg_pairwise_corr(win[cols])
        vol_state[d] = win[cols].mean(axis=1).std() * np.sqrt(252)
    corr_state = pd.Series(corr_state)
    vol_state = pd.Series(vol_state)

    def terciles(cond: pd.Series, name: str) -> None:
        common_idx = m.index.intersection(cond.dropna().index)
        mm, cc = m.loc[common_idx], cond.loc[common_idx]
        q1, q2 = cc.quantile([1 / 3, 2 / 3])
        buckets = {
            f"{name} BAIXO":  cc <= q1,
            f"{name} MÉDIO":  (cc > q1) & (cc <= q2),
            f"{name} ALTO":   cc > q2,
        }
        for label, mask in buckets.items():
            seg = mm[mask.values]
            line(label, seg, seg[seg.index.year != 2020])

    print("\n" + "═" * 74)
    print("B) CONDICIONAMENTO POINT-IN-TIME (tercil medido no INÍCIO do mês)")
    print("═" * 74)
    print(hdr)
    print("-" * 74)
    terciles(corr_state, "corr média")
    print("-" * 74)
    terciles(vol_state, "vol mercado")


if __name__ == "__main__":
    main()
