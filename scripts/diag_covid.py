"""
scripts/diag_covid.py — o alpha do GDS é só o evento COVID (2020)?

Decompõe o gross L/S do GDS por ano-calendário e re-mede o alpha removendo
a janela COVID. Um sinal geométrico (curvatura de correlações) é o suspeito
nº1 de ter todo o retorno no choque de mar/2020 + recuperação. Se ao tirar
2020 (ou só o crash) o alpha some, não é sinal — é um único evento.
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


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {"n": len(r), "ret_aa": np.nan, "sharpe": np.nan, "t": np.nan}
    mu, sd = r.mean(), r.std()
    return {
        "n": len(r),
        "ret_aa": round(mu * 252, 4),
        "sharpe": round(mu / sd * np.sqrt(252), 3),
        "t": round(mu / (sd / np.sqrt(len(r))), 2),
    }


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
    eng.oos_start = gds.index[0]            # amostra inteira
    y = eng.run_continuous(returns=returns, gds_ts=gds, prices=prices)["gross"]

    print("\n" + "═" * 60)
    print("GROSS L/S GDS — decomposição por ano")
    print("═" * 60)
    print(f"{'ano':<8}{'n':>6}{'ret_aa':>10}{'sharpe':>9}{'t':>7}")
    for yr, seg in y.groupby(y.index.year):
        s = stats(seg)
        print(f"{yr:<8}{s['n']:>6}{s['ret_aa']:>10.2%}{s['sharpe']:>9}{s['t']:>7}")

    # Janelas a remover
    covid_crash = (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"))
    masks = {
        "amostra cheia":        pd.Series(True, index=y.index),
        "sem 2020 inteiro":     y.index.year != 2020,
        "sem crash (fev–jun20)": ~((y.index >= covid_crash[0]) & (y.index <= covid_crash[1])),
        "só OOS 2019+":         y.index.year >= 2019,
        "OOS sem 2020":         (y.index.year >= 2019) & (y.index.year != 2020),
    }
    print("\n" + "═" * 60)
    print("ALPHA do gross sob diferentes recortes")
    print("═" * 60)
    print(f"{'recorte':<24}{'n':>6}{'ret_aa':>10}{'sharpe':>9}{'t':>7}")
    for label, m in masks.items():
        s = stats(y[m])
        print(f"{label:<24}{s['n']:>6}{s['ret_aa']:>10.2%}{s['sharpe']:>9}{s['t']:>7}")

    # Quanto de 2020 pesa no acumulado total (soma de log-returns)
    total = y.sum()
    y2020 = y[y.index.year == 2020].sum()
    crash = y[(y.index >= covid_crash[0]) & (y.index <= covid_crash[1])].sum()
    print("\n" + "─" * 60)
    print(f"Retorno log acumulado TOTAL          : {total:+.3f}")
    print(f"  contribuição de 2020              : {y2020:+.3f}  ({y2020/total:.0%} do total)")
    print(f"  contribuição do crash (fev–jun20) : {crash:+.3f}  ({crash/total:.0%} do total)")


if __name__ == "__main__":
    main()
