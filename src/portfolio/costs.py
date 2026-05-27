"""
src/portfolio/costs.py
──────────────────────
Modelagem de custos de transação diferenciados por grupo de liquidez.

ESTRUTURA DE CUSTOS (conforme settings.yaml)
─────────────────────────────────────────────
  Large caps  (VALE, PETR, ITUB, ...): comissão 8bps + spread 5bps = 13bps RT
  Mid caps    (restante do Ibovespa) : comissão 10bps + spread 12bps = 22bps RT
  Small liquid (fora do Ibov, >R$5M): comissão 12bps + spread 20bps = 32bps RT
  Custo de aluguel short             : ~3% ao ano = ~0.012% ao dia

RT = round-trip (comprar + vender)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.data.config import get


class TransactionCostModel:
    """
    Modela custos de transação diferenciados por grupo de liquidez.

    Os custos são aplicados sobre o turnover — apenas a parcela
    do portfólio que muda entre rebalanceamentos gera custo.

    Uso típico
    ----------
    cost_model = TransactionCostModel()
    net_returns = cost_model.apply(gross_returns, weights_df, returns_df)
    """

    def __init__(self) -> None:
        cfg = get("costs")
        self.large_caps = cfg.get("large_caps", {})
        self.mid_caps   = cfg.get("mid_caps", {})
        self.small_liq  = cfg.get("small_liquid", {})
        self.short_borrow_annual = cfg.get("short_borrow_annual_pct", 3.0) / 100.0

        # Tickers de large caps (da config)
        self.large_cap_tickers = set(self.large_caps.get("tickers", []))

    # ──────────────────────────────────────────────────────────────
    # INTERFACE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def apply(
        self,
        gross_returns: pd.Series,
        weights_df: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.Series:
        """
        Aplica custos de transação para obter retornos líquidos.

        gross_returns(t) = Σ_i w_i(t-1) * r_i(t)
        custo(t)         = custo_transação(t) + custo_aluguel(t)
        net_returns(t)   = gross_returns(t) - custo(t)

        Parameters
        ----------
        gross_returns : retornos brutos do portfólio (Serie por data)
        weights_df    : pesos do portfólio por data e ativo
        prices        : preços para estimar volume de negociação

        Returns
        -------
        pd.Series de retornos líquidos de custos.
        """
        transaction_costs = self._compute_transaction_costs(weights_df)
        borrow_costs      = self._compute_borrow_costs(weights_df)

        total_costs = (transaction_costs + borrow_costs).reindex(gross_returns.index).fillna(0)

        net = gross_returns - total_costs

        logger.info(
            f"Custos de transação:\n"
            f"  Custo médio por rebalanceamento : {transaction_costs.mean():.4%}\n"
            f"  Custo de aluguel médio/dia      : {borrow_costs.mean():.6%}\n"
            f"  Drag total anualizado           : {total_costs.mean() * 252:.2%}"
        )
        return net

    # ──────────────────────────────────────────────────────────────
    # CÁLCULO DE CUSTOS
    # ──────────────────────────────────────────────────────────────
    def _compute_transaction_costs(
        self,
        weights_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Custo de transação = Σ_i |Δw_i(t)| * cost_rate_i

        O custo_rate por ativo depende do grupo de liquidez.
        É aplicado apenas nas datas de rebalanceamento.
        """
        # Turnover por ativo
        delta_w = weights_df.diff().abs()
        delta_w.iloc[0] = weights_df.iloc[0].abs()  # primeiro rebal

        # Custo por ativo (em bps, convertido para decimal)
        cost_rates = self._get_cost_rates(weights_df.columns)

        # Custo = turnover × taxa
        costs_per_asset = delta_w * cost_rates
        total_cost = costs_per_asset.sum(axis=1)

        return total_cost

    def _compute_borrow_costs(
        self,
        weights_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Custo de aluguel de ações para posições short.

        borrow_cost(t) = |Σ_{i short} w_i(t)| × (borrow_rate / 252)

        Aplicado a cada dia útil, não apenas nos rebalanceamentos.
        """
        short_exposure = weights_df[weights_df < 0].abs().sum(axis=1)
        daily_borrow_rate = self.short_borrow_annual / 252.0
        return short_exposure * daily_borrow_rate

    def _get_cost_rates(self, tickers: pd.Index) -> pd.Series:
        """
        Retorna a taxa de custo total (comissão + spread) por ativo.
        Round-trip em decimal (já inclui ida e volta).
        """
        rates = {}
        for ticker in tickers:
            if ticker in self.large_cap_tickers:
                # Large cap: 8bps comissão + 5bps spread = 13bps RT
                comm = self.large_caps.get("commission_bps", 8)
                spread = self.large_caps.get("spread_bps", 5)
            else:
                # Mid cap: 10bps comissão + 12bps spread = 22bps RT
                comm = self.mid_caps.get("commission_bps", 10)
                spread = self.mid_caps.get("spread_bps", 12)

            # RT cost = (comissão + metade do spread) × 2
            rt_bps = (comm + spread / 2.0) * 2
            rates[ticker] = rt_bps / 10_000.0  # bps → decimal

        return pd.Series(rates)

    def get_cost_summary(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Relatório de custos por grupo de ativo.
        Útil para o relatório final.
        """
        cost_rates = self._get_cost_rates(weights_df.columns)
        turnover = weights_df.diff().abs()
        turnover.iloc[0] = weights_df.iloc[0].abs()

        avg_turnover = turnover.mean()
        avg_cost = (turnover * cost_rates).mean()

        df = pd.DataFrame({
            "grupo": [
                "large_cap" if t in self.large_cap_tickers else "mid_cap"
                for t in weights_df.columns
            ],
            "taxa_rt_bps":      (cost_rates * 10_000).values,
            "turnover_medio":   avg_turnover.values,
            "custo_medio":      avg_cost.values,
        }, index=weights_df.columns)

        return df.groupby("grupo").agg({
            "taxa_rt_bps": "mean",
            "turnover_medio": "mean",
            "custo_medio": "sum",
        }).round(4)
