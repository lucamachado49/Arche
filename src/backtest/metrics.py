"""
src/backtest/metrics.py
───────────────────────
Métricas de performance do portfólio.

Todas as métricas são computadas ex-post sobre a série de retornos.
NUNCA são usadas para tomar decisões de alocação — apenas para avaliação. ✓
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class PerformanceMetrics:
    """
    Calcula métricas de performance para uma série de retornos.

    Uso típico
    ----------
    pm = PerformanceMetrics(returns=net_returns, benchmark=ibov_returns)
    summary = pm.summary()
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark: pd.Series | None = None,
        risk_free: pd.Series | None = None,
        periods_per_year: int = 252,
    ) -> None:
        self.returns  = returns.dropna()
        self.benchmark = benchmark.reindex(self.returns.index).dropna() if benchmark is not None else None
        self.rf       = risk_free.reindex(self.returns.index).fillna(0) if risk_free is not None else None
        self.ann_factor = periods_per_year

    # ──────────────────────────────────────────────────────────────
    # MÉTRICAS PRINCIPAIS
    # ──────────────────────────────────────────────────────────────
    def annualized_return(self) -> float:
        """Retorno anualizado geométrico."""
        total = (1 + self.returns).prod()
        n = len(self.returns)
        return float(total ** (self.ann_factor / n) - 1)

    def annualized_vol(self) -> float:
        """Volatilidade anualizada."""
        return float(self.returns.std() * np.sqrt(self.ann_factor))

    def sharpe_ratio(self) -> float:
        """Sharpe Ratio (excesso sobre risk-free, anualizado)."""
        if self.rf is not None:
            excess = self.returns - self.rf.reindex(self.returns.index).fillna(0)
        else:
            excess = self.returns

        mean_excess = excess.mean() * self.ann_factor
        vol = excess.std() * np.sqrt(self.ann_factor)

        return float(mean_excess / vol) if vol > 1e-9 else 0.0

    def sortino_ratio(self, mar: float = 0.0) -> float:
        """Sortino Ratio — penaliza apenas a volatilidade negativa."""
        downside = self.returns[self.returns < mar]
        downside_vol = downside.std() * np.sqrt(self.ann_factor)
        ann_ret = self.annualized_return()
        return float((ann_ret - mar) / downside_vol) if downside_vol > 1e-9 else 0.0

    def max_drawdown(self) -> float:
        """Máximo drawdown (peak-to-trough)."""
        cum = (1 + self.returns).cumprod()
        rolling_max = cum.cummax()
        drawdowns = cum / rolling_max - 1
        return float(drawdowns.min())

    def calmar_ratio(self) -> float:
        """Calmar Ratio = retorno anualizado / |max drawdown|."""
        mdd = abs(self.max_drawdown())
        if mdd < 1e-9:
            return 0.0
        return float(self.annualized_return() / mdd)

    def win_rate(self) -> float:
        """Percentual de dias/períodos com retorno positivo."""
        return float((self.returns > 0).mean())

    def skewness(self) -> float:
        """Assimetria da distribuição de retornos."""
        return float(stats.skew(self.returns.dropna()))

    def kurtosis(self) -> float:
        """Excesso de curtose da distribuição de retornos."""
        return float(stats.kurtosis(self.returns.dropna()))

    def var_95(self) -> float:
        """Value at Risk a 95% de confiança (histórico)."""
        return float(np.percentile(self.returns, 5))

    def cvar_95(self) -> float:
        """Conditional VaR (Expected Shortfall) a 95%."""
        threshold = self.var_95()
        tail = self.returns[self.returns <= threshold]
        return float(tail.mean()) if len(tail) > 0 else threshold

    def information_ratio(self) -> float | None:
        """Information Ratio vs benchmark."""
        if self.benchmark is None:
            return None
        common = self.returns.index.intersection(self.benchmark.index)
        if len(common) < 20:
            return None
        active = self.returns.loc[common] - self.benchmark.loc[common]
        ir = active.mean() / active.std() * np.sqrt(self.ann_factor)
        return float(ir)

    def t_stat_return(self) -> float:
        """t-statistic para testar H0: retorno médio = 0."""
        n = len(self.returns)
        mean = self.returns.mean()
        se = self.returns.std() / np.sqrt(n)
        return float(mean / se) if se > 1e-9 else 0.0

    # ──────────────────────────────────────────────────────────────
    # RELATÓRIO COMPLETO
    # ──────────────────────────────────────────────────────────────
    def summary(self) -> pd.Series:
        """Retorna todas as métricas em uma Series."""
        d = {
            "Retorno Anualizado":    f"{self.annualized_return():.2%}",
            "Volatilidade Anual":    f"{self.annualized_vol():.2%}",
            "Sharpe Ratio":          f"{self.sharpe_ratio():.4f}",
            "Sortino Ratio":         f"{self.sortino_ratio():.4f}",
            "Calmar Ratio":          f"{self.calmar_ratio():.4f}",
            "Max Drawdown":          f"{self.max_drawdown():.2%}",
            "Win Rate":              f"{self.win_rate():.1%}",
            "VaR 95% (diário)":      f"{self.var_95():.4%}",
            "CVaR 95% (diário)":     f"{self.cvar_95():.4%}",
            "Skewness":              f"{self.skewness():.4f}",
            "Kurtosis (excesso)":    f"{self.kurtosis():.4f}",
            "t-stat retorno":        f"{self.t_stat_return():.4f}",
            "N observações":         len(self.returns),
            "Período":               f"{self.returns.index[0].date()} → {self.returns.index[-1].date()}",
        }
        if self.benchmark is not None:
            ir = self.information_ratio()
            d["Information Ratio"] = f"{ir:.4f}" if ir is not None else "N/A"

        return pd.Series(d, name="Métricas de Performance")

    def rolling_sharpe(self, window: int = 252) -> pd.Series:
        """Sharpe rolling anualizado (útil para análise de consistência)."""
        rolling_mean = self.returns.rolling(window).mean() * self.ann_factor
        rolling_std  = self.returns.rolling(window).std() * np.sqrt(self.ann_factor)
        return (rolling_mean / rolling_std).replace([np.inf, -np.inf], np.nan)

    def drawdown_series(self) -> pd.Series:
        """Série temporal de drawdown (útil para gráfico)."""
        cum = (1 + self.returns).cumprod()
        rolling_max = cum.cummax()
        return cum / rolling_max - 1
