"""
scripts/diag_tilt.py — bancada do Enhanced Index (tilt long-only) com buffering.

Testa a hipótese: abandonar o L/S isolado e usar o GDS como TILT sobre um
benchmark, long-only, com banda de tolerância (buffering). Aqui o buffering
MORDE — os pesos são ancorados no benchmark (sticky) e o aluguel de short
some. Métrica que importa: alpha sobre o benchmark (active return), Tracking
Error e Information Ratio — não o Sharpe absoluto.

Benchmark: equal-weight do universo do GDS (proxy do índice; produção usaria
os pesos float-cap reais do IBrX-100). A mecânica tilt+buffer independe disso.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()

from src.data.config import PROJECT_ROOT, get
from src.data.cleaner import DataCleaner
from src.signal.generator import SignalGenerator
from src.portfolio.costs import TransactionCostModel


SPAN = get("signal", "gds_smoothing_span", 20)
SIGN = get("signal", "signal_sign", 1)
OOS = pd.Timestamp(get("backtest", "out_of_sample_start", "2019-01-01"))


def monthly(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def tilt_weights(z: pd.Series, budget: float, prev: pd.Series | None, buffer: float) -> pd.Series:
    """Long-only enhanced index: w = (w_bench + a·z̃)₊, renormalizado, com buffering."""
    names = z.index
    w_bench = pd.Series(1.0 / len(names), index=names)
    zz = z - z.mean()
    scale = zz.abs().sum() / 2.0
    if scale < 1e-12:
        target = w_bench.copy()
    else:
        a = budget / scale                       # budget = overweight total (= underweight)
        target = (w_bench + a * zz).clip(lower=0.0)
        target = target / target.sum()
    if prev is not None and buffer > 0:
        pr = prev.reindex(names).fillna(0.0)
        hold = (target - pr).abs() < buffer
        target[hold] = pr[hold]
        target = target / target.sum()
    return target


def run(gds_m: pd.DataFrame, returns: pd.DataFrame, prices: pd.DataFrame,
        budget: float, buffer: float) -> dict:
    cost_model = TransactionCostModel()
    weights = {}
    prev = None
    for d in gds_m.index:
        z = (SIGN * gds_m.loc[d]).dropna()
        if len(z) < 10:
            continue
        w = tilt_weights(z, budget, prev, buffer)
        weights[d] = w
        prev = w
    W = pd.DataFrame(weights).T.fillna(0.0)

    # Retorno do fundo (look-ahead safe: peso de ontem × retorno de hoje)
    common = [c for c in W.columns if c in returns.columns]
    Wd = W[common].reindex(returns.index).ffill()
    fund_gross = (returns[common] * Wd.shift(1)).sum(axis=1)
    fund_gross = fund_gross.loc[fund_gross.index >= W.index[0]].dropna()

    net = cost_model.apply(fund_gross, W[common], prices)  # long-only ⇒ borrow=0

    # Benchmark EW do mesmo universo
    bench = returns[common].reindex(fund_gross.index).mean(axis=1)

    act_g = (fund_gross - bench).dropna()
    act_n = (net.reindex(fund_gross.index) - bench).dropna()
    te = act_n.std() * np.sqrt(252)
    ir_g = act_g.mean() / act_g.std() * np.sqrt(252)
    ir_n = act_n.mean() / act_n.std() * np.sqrt(252)
    per_rebal = W[common].diff().abs().sum(axis=1).iloc[1:] / 2.0
    turn = per_rebal.mean()
    n_active = int((per_rebal > 0.005).sum())   # rebalances que realmente negociam
    # Estabilidade: alpha líquido nas duas metades do OOS (anualizado)
    half = act_n.index[len(act_n) // 2]
    a1 = act_n.loc[:half].mean() * 252
    a2 = act_n.loc[half:].mean() * 252
    return {
        "budget": budget, "buffer": buffer,
        "alpha_líq_aa": round(act_n.mean() * 252, 4),
        "TE_aa": round(te, 4),
        "IR_bruto": round(ir_g, 3),
        "IR_líq": round(ir_n, 3),
        "turn/reb": round(turn, 3),
        "n_ativo": n_active,
        "α_1ªmet": round(a1, 4),
        "α_2ªmet": round(a2, 4),
    }


def main() -> None:
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    cache_dir = PROJECT_ROOT / "data" / "cache" / "transfer" / "BR"
    _, gds = SignalGenerator(cache_dir=cache_dir).run(prices=prices, use_cache=True)

    common = [c for c in gds.columns if c in prices.columns]
    prices = prices[common]
    returns = DataCleaner().prices_to_log_returns(prices)

    gds_s = gds.ewm(span=SPAN, adjust=False).mean()
    gds_oos = gds_s.loc[gds_s.index >= OOS]
    gds_m = gds_oos.loc[gds_oos.index.isin(monthly(gds_oos.index))]

    rows = []
    for budget in (0.10, 0.20, 0.30):
        for buffer in (0.0, 0.005, 0.01, 0.02):
            rows.append(run(gds_m, returns, prices, budget, buffer))

    df = pd.DataFrame(rows)
    print("\n" + "═" * 78)
    print(f"ENHANCED INDEX (tilt long-only) — EWMA span={SPAN}, sign={SIGN}, bench=EW")
    print("alpha/TE/IR anualizados; budget = over/under-weight total alocado pelo sinal")
    print("═" * 78)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
