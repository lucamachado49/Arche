"""
scripts/diag_recon_backtest.py — o sinal reconstruído (diff4) muda a economia?

O IC ex-2020 do Δκ de 4 semanas (diff4) sobrevive com t=2.22 (vs t=1.20 do
diff1 atual). Pergunta decisiva: isso vira retorno LÍQUIDO de custo, ou o IC
ainda é fraco demais? Roda diff4 no engine contínuo (L/S) e no tilt long-only,
full e ex-2020, comparando com o sinal atual.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()

from src.data.config import PROJECT_ROOT, get
from src.data.cleaner import DataCleaner
from src.backtest.walkforward import WalkForwardBacktest
from src.portfolio.costs import TransactionCostModel

SIGN = get("signal", "signal_sign", 1)


def dstats(r):
    r = r.dropna()
    if len(r) < 20:
        return dict(ret=np.nan, sh=np.nan, t=np.nan)
    return dict(ret=r.mean() * 252, sh=r.mean() / r.std() * np.sqrt(252),
                t=r.mean() / (r.std() / np.sqrt(len(r))))


def line(label, r):
    a, ex = dstats(r), dstats(r[r.index.year != 2020])
    print(f"{label:<32}{a['ret']:>8.2%}{a['sh']:>7.2f}{a['t']:>6.2f}  | "
          f"{ex['ret']:>8.2%}{ex['t']:>6.2f}")


def tilt_series(gds_m, returns, cost):
    weights, prev = {}, None
    for d in gds_m.index:
        z = (SIGN * gds_m.loc[d]).dropna()
        if len(z) < 10:
            continue
        wb = pd.Series(1.0 / len(z), index=z.index)
        zz = z - z.mean(); s = zz.abs().sum() / 2.0
        tgt = wb.copy() if s < 1e-12 else (wb + (0.20 / s) * zz).clip(lower=0.0)
        tgt = tgt / tgt.sum()
        if prev is not None:
            pr = prev.reindex(tgt.index).fillna(0.0)
            hold = (tgt - pr).abs() < 0.01
            tgt[hold] = pr[hold]; tgt = tgt / tgt.sum()
        weights[d] = tgt; prev = tgt
    W = pd.DataFrame(weights).T.fillna(0.0)
    cc = [c for c in W.columns if c in returns.columns]
    Wd = W[cc].reindex(returns.index).ffill()
    fund = (returns[cc] * Wd.shift(1)).sum(axis=1)
    fund = fund.loc[fund.index >= W.index[0]]
    net = cost.apply(fund, W[cc], prices)
    bench = returns[cc].reindex(fund.index).mean(axis=1)
    return (net.reindex(fund.index) - bench).dropna()


prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
prices.index = pd.to_datetime(prices.index); prices = prices.sort_index()
kappa = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "transfer" / "BR" / "kappa_ts.parquet")
gds_old = pd.read_parquet(PROJECT_ROOT / "data" / "cache" / "transfer" / "BR" / "gds_ts.parquet")
kappa.index = pd.to_datetime(kappa.index); gds_old.index = pd.to_datetime(gds_old.index)
common = [c for c in kappa.columns if c in prices.columns]
kappa, gds_old = kappa[common], gds_old[common]
returns = DataCleaner().prices_to_log_returns(prices[common])
cost = TransactionCostModel()

sigma = kappa.expanding(min_periods=24).std()
gds_diff4 = (kappa.diff(4) / sigma).dropna(how="all")

hdr = f"{'config':<32}{'ret_aa':>8}{'sh':>7}{'t':>6}  | {'ex20':>8}{'t':>6}"


def run_ls(gds_sig, span, label):
    eng = WalkForwardBacktest()
    eng.oos_start = gds_sig.index[0]
    eng.gds_smoothing_span = span
    out = eng.run_continuous(returns=returns, gds_ts=gds_sig, prices=prices)
    line(label, out["net"])


print("\n" + "═" * 70)
print("L/S QUINTIL LÍQUIDO — sinal atual vs reconstruído (diff4)")
print("═" * 70); print(hdr); print("-" * 70)
run_ls(gds_old, 20, "diff1+ewma20 (atual)")
run_ls(gds_diff4, 1, "diff4 (sem suavização extra)")
run_ls(gds_diff4, 6, "diff4 + ewma6")

print("\n" + "═" * 70)
print("ENHANCED INDEX TILT — alpha líquido vs benchmark EW")
print("═" * 70); print(hdr); print("-" * 70)
line("diff1+ewma20 (atual)",
     tilt_series(gds_old.ewm(span=20, adjust=False).mean().loc[gds_old.index.isin(
         WalkForwardBacktest().run_continuous(returns=returns, gds_ts=gds_old, prices=prices)["weights"].index)],
         returns, cost))
gds4_s = gds_diff4.ewm(span=6, adjust=False).mean()
mdates = pd.DatetimeIndex(gds4_s.index.to_series().groupby([gds4_s.index.year, gds4_s.index.month]).max().values)
line("diff4 + ewma6", tilt_series(gds4_s.loc[gds4_s.index.isin(mdates)], returns, cost))
