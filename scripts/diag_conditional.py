"""
scripts/diag_conditional.py — GDS condicional ao regime de correlação.

Tese (de diag_regime): o GDS paga quando a correlação média do mercado está
alta. Aqui torna-se FALSIFICÁVEL e tradeable: só carrega o book quando a
correlação média par-a-par (trailing 60d, point-in-time) está acima do seu
quantil 2/3 EXPANSÍVEL (limiar usa só o passado; warmup 24 meses). Mede
líquido de custo em duas estruturas e compara com a versão always-on:

  • L/S quintil   (sinal mais forte, custo alto)
  • Enhanced index long-only (tilt+buffer): off ⇒ segura o benchmark, só
    desvia em regime de correlação alta. Produto institucional natural.

Reporta amostra cheia, ex-2020 e split-half (anti-snooping).
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
from src.signal.generator import SignalGenerator
from src.backtest.walkforward import WalkForwardBacktest
from src.portfolio.costs import TransactionCostModel

SIGN = get("signal", "signal_sign", 1)
SPAN = get("signal", "gds_smoothing_span", 20)
WARMUP = 24          # meses de história p/ estimar o limiar expansível
Q = 2 / 3            # tercil alto


def month_ends(idx):
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def avg_pairwise_corr(win):
    w = win.dropna(axis=1, thresh=int(0.8 * len(win)))
    if w.shape[1] < 5:
        return np.nan
    c = w.corr().values
    n = c.shape[0]
    return (np.nansum(c) - n) / (n * (n - 1))


def dstats(r):
    r = r.dropna()
    if len(r) < 20:
        return dict(ret=np.nan, sh=np.nan, t=np.nan)
    return dict(ret=r.mean() * 252, sh=r.mean() / r.std() * np.sqrt(252),
                t=r.mean() / (r.std() / np.sqrt(len(r))))


def tilt_w(z, budget, prev, buffer):
    names = z.index
    wb = pd.Series(1.0 / len(names), index=names)
    zz = z - z.mean()
    s = zz.abs().sum() / 2.0
    tgt = wb.copy() if s < 1e-12 else (wb + (budget / s) * zz).clip(lower=0.0)
    tgt = tgt / tgt.sum()
    if prev is not None and buffer > 0:
        pr = prev.reindex(names).fillna(0.0)
        hold = (tgt - pr).abs() < buffer
        tgt[hold] = pr[hold]
        tgt = tgt / tgt.sum()
    return tgt


def report(label, r):
    a = dstats(r)
    ex = dstats(r[r.index.year != 2020])
    half = r.index[len(r) // 2]
    h1 = dstats(r.loc[:half]); h2 = dstats(r.loc[half:])
    print(f"{label:<30}{a['ret']:>8.2%}{a['sh']:>7.2f}{a['t']:>6.2f}  | "
          f"{ex['ret']:>8.2%}{ex['t']:>6.2f}  | {h1['ret']:>8.2%}{h2['ret']:>8.2%}")


def main():
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet")
    prices.index = pd.to_datetime(prices.index); prices = prices.sort_index()
    cache = PROJECT_ROOT / "data" / "cache" / "transfer" / "BR"
    _, gds = SignalGenerator(cache_dir=cache).run(prices=prices, use_cache=True)
    common = [c for c in gds.columns if c in prices.columns]
    prices, gds = prices[common], gds[common]
    returns = DataCleaner().prices_to_log_returns(prices)
    cost = TransactionCostModel()

    eng = WalkForwardBacktest(); eng.oos_start = gds.index[0]
    out = eng.run_continuous(returns=returns, gds_ts=gds, prices=prices)
    W = out["weights"]                                # pesos L/S mensais

    # ── Regime: correlação média + limiar expansível point-in-time ──
    corr = {}
    for d in W.index:
        win = returns.loc[returns.index <= d].tail(60)
        if len(win) >= 40:
            corr[d] = avg_pairwise_corr(win[common])
    corr = pd.Series(corr).sort_index()
    thr = corr.expanding(min_periods=WARMUP).quantile(Q).shift(1)
    in_mkt = (corr > thr).reindex(W.index).fillna(False)
    print(f"\nEm mercado: {in_mkt.mean():.0%} dos meses "
          f"({int(in_mkt.sum())}/{len(in_mkt)}) | limiar expansível q={Q:.2f}")

    hdr = (f"{'estratégia':<30}{'ret_aa':>8}{'sh':>7}{'t':>6}  | "
           f"{'ex20_ret':>8}{'t':>6}  | {'1ªmet':>8}{'2ªmet':>8}")
    print("\n" + "═" * 88)
    print("L/S QUINTIL — always-on vs condicional à correlação (líquido de custo)")
    print("═" * 88); print(hdr); print("-" * 88)

    # always-on
    gross_b = out["gross"]; net_b = out["net"]
    report("L/S always-on (gross)", gross_b)
    report("L/S always-on (net)", net_b)

    # condicional: zera o book fora do regime
    Wc = W.copy()
    Wc.loc[~in_mkt.values] = 0.0
    gross_c = eng._compute_gross_returns(returns, Wc)
    gross_c = gross_c.loc[gross_c.index >= W.index[0]]
    net_c = cost.apply(gross_c, Wc, prices)
    report("L/S condicional (gross)", gross_c)
    report("L/S condicional (net)", net_c)

    # ── Enhanced index tilt (long-only): off ⇒ benchmark ──
    print("\n" + "═" * 88)
    print("ENHANCED INDEX (tilt long-only, budget=0.20, buffer=0.01) — ALPHA vs benchmark EW")
    print("═" * 88); print(hdr); print("-" * 88)

    gds_s = gds.ewm(span=SPAN, adjust=False).mean()
    gds_m = gds_s.loc[gds_s.index.isin(W.index)]
    for cond in (False, True):
        weights, prev = {}, None
        for d in gds_m.index:
            z = (SIGN * gds_m.loc[d]).dropna()
            if len(z) < 10:
                continue
            wb = pd.Series(1.0 / len(z), index=z.index)
            if cond and not bool(in_mkt.get(d, False)):
                w = wb                       # fora do regime: segura o índice
            else:
                w = tilt_w(z, 0.20, prev, 0.01)
            weights[d] = w; prev = w
        Wt = pd.DataFrame(weights).T.fillna(0.0)
        cc = [c for c in Wt.columns if c in returns.columns]
        Wd = Wt[cc].reindex(returns.index).ffill()
        fund = (returns[cc] * Wd.shift(1)).sum(axis=1)
        fund = fund.loc[fund.index >= Wt.index[0]]
        net = cost.apply(fund, Wt[cc], prices)
        bench = returns[cc].reindex(fund.index).mean(axis=1)
        act = (net.reindex(fund.index) - bench).dropna()
        report(f"tilt {'CONDICIONAL' if cond else 'always-on'} (alpha líq)", act)


if __name__ == "__main__":
    main()
