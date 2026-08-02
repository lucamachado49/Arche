"""
scripts/diag_tier1.py — diagnóstico controlado das alavancas do Tier 1.

Isola o efeito de (método de construção × banda de no-trade) sobre o
Sharpe bruto/líquido e o turnover, reaproveitando o cache de GDS. Não
substitui os relatórios oficiais — é uma bancada de experimentos.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()  # silencia o ruído de INFO p/ o diagnóstico

from src.data.config import PROJECT_ROOT
from src.data.cleaner import DataCleaner
from src.signal.generator import SignalGenerator
from src.backtest.walkforward import WalkForwardBacktest
from src.backtest.metrics import PerformanceMetrics


def main() -> None:
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    cache_dir = PROJECT_ROOT / "data" / "cache" / "transfer" / "BR"
    _, gds_ts = SignalGenerator(cache_dir=cache_dir).run(prices=prices, use_cache=True)

    common = [c for c in gds_ts.columns if c in prices.columns]
    prices = prices[common]
    returns = DataCleaner().prices_to_log_returns(prices)

    # Engine CONTÍNUO (sem artefato de recompra/fold). Varre:
    #   método × span EWMA × exit_quantile (hysteresis; 0.20 = quintil puro)
    combos = [
        # método,   ewma, entry, exit
        ("quintile", 20, 0.20, 0.20),   # quintil puro
        ("quintile", 20, 0.20, 0.30),
        ("quintile", 20, 0.20, 0.35),
        ("quintile", 20, 0.20, 0.50),
        ("quintile", 26, 0.20, 0.20),
        ("quintile", 26, 0.20, 0.40),
        ("quintile",  1, 0.20, 0.20),   # sem suavização (referência)
        ("zscore",   20, 0.20, 0.20),
    ]

    rows = []
    for method, span, entry, exit_ in combos:
        engine = WalkForwardBacktest()
        engine.gds_smoothing_span = span
        engine.constructor.method = method
        engine.constructor.no_trade_band = 0.0
        engine.constructor.entry_quantile = entry
        engine.constructor.exit_quantile = exit_

        out = engine.run_continuous(returns=returns, gds_ts=gds_ts, prices=prices)
        pm_g, pm_n = PerformanceMetrics(out["gross"]), PerformanceMetrics(out["net"])
        turn = engine.constructor.get_turnover(out["weights"])

        rows.append({
            "método": method,
            "ewma": span,
            "exit_q": exit_,
            "sharpe_bruto": round(pm_g.sharpe_ratio(), 3),
            "sharpe_líq": round(pm_n.sharpe_ratio(), 3),
            "ret_líq_aa": round(pm_n.annualized_return(), 4),
            "turn/rebal": round(float(turn.iloc[1:].mean()), 3),
        })

    df = pd.DataFrame(rows)
    print("\n" + "═" * 70)
    print("DIAGNÓSTICO TIER 1 — método × banda (rebalance mensal, signal_sign=+1)")
    print("═" * 70)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
