"""
src/portfolio/constructor.py
────────────────────────────
Construção do portfólio long-short baseado no GDS.

ESTRATÉGIA
──────────
1. Rankear ativos por GDS em cada data de rebalanceamento
2. Quintil Q1 (GDS mais negativo) → LONG  (hipótese: reversão geométrica)
3. Quintil Q5 (GDS mais positivo) → SHORT (hipótese: sobre-extensão geométrica)
4. Pesos: igual-ponderados dentro de cada perna, sujeito a max_weight
5. Portfólio market-neutral: valor long = valor short

LOOK-AHEAD SAFETY
──────────────────
Os pesos em t são calculados apenas com GDS(t) disponível em t.
O retorno do portfólio é realizado de t+1 em diante. ✓
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data.config import get


class PortfolioConstructor:
    """
    Constrói o portfólio long-short a partir do sinal GDS.

    Uso típico
    ----------
    constructor = PortfolioConstructor()
    weights = constructor.build_weights(gds_t, eligible_tickers)
    # weights: Series com pesos positivos (long) e negativos (short)
    """

    def __init__(self) -> None:
        cfg = get("portfolio")
        self.n_quantiles    = cfg.get("n_quantiles", 5)
        self.max_weight     = cfg.get("max_weight_single_asset", 0.15)
        self.long_short_ratio = cfg.get("long_short_ratio", 1.0)

    # ──────────────────────────────────────────────────────────────
    # CONSTRUÇÃO DE PESOS
    # ──────────────────────────────────────────────────────────────
    def build_weights(
        self,
        gds_t: pd.Series,
        eligible_tickers: Optional[list[str]] = None,
    ) -> pd.Series:
        """
        Constrói pesos do portfólio para uma data de rebalanceamento.

        Estratégia quintil:
          - Q1 (GDS mais baixo): LONG com peso > 0
          - Q5 (GDS mais alto): SHORT com peso < 0
          - Q2, Q3, Q4: fora do portfólio (peso = 0)

        O portfólio é market-neutral: Σ w_long = 1, Σ w_short = -1
        (ou ajustado pelo long_short_ratio).

        Parameters
        ----------
        gds_t           : GDS na data t (Series, index=tickers)
        eligible_tickers : filtro adicional de elegibilidade

        Returns
        -------
        pd.Series com pesos (soma(positivos) ≈ 1, soma(negativos) ≈ -1)
        """
        # Filtrar para universo elegível
        if eligible_tickers is not None:
            gds_t = gds_t.reindex(eligible_tickers).dropna()
        else:
            gds_t = gds_t.dropna()

        if len(gds_t) < self.n_quantiles * 2:
            logger.warning(
                f"Apenas {len(gds_t)} ativos disponíveis. "
                f"Mínimo para {self.n_quantiles} quintis: {self.n_quantiles * 2}."
            )
            return pd.Series(dtype=float)

        # Rankear por GDS
        ranked = gds_t.rank(method="average")
        n = len(ranked)
        quantile_size = n / self.n_quantiles

        # Definir quintis
        q1_threshold = quantile_size         # bottom quintil = LONG
        q5_threshold = n - quantile_size     # top quintil = SHORT

        long_tickers  = ranked[ranked <= q1_threshold].index.tolist()
        short_tickers = ranked[ranked > q5_threshold].index.tolist()

        if not long_tickers or not short_tickers:
            logger.warning("Quintis long/short vazios — portfólio não construído.")
            return pd.Series(dtype=float)

        # Pesos iguais dentro de cada perna
        weights = pd.Series(0.0, index=gds_t.index)

        w_long  = 1.0 / len(long_tickers)
        w_short = -self.long_short_ratio / len(short_tickers)

        weights[long_tickers]  = w_long
        weights[short_tickers] = w_short

        # Aplicar cap de peso máximo
        weights = self._apply_weight_cap(weights)

        # Renormalizar após o cap
        weights = self._renormalize(weights)

        logger.debug(
            f"Portfólio: {len(long_tickers)} long, {len(short_tickers)} short | "
            f"w_long={weights[weights>0].sum():.2f}, w_short={weights[weights<0].sum():.2f}"
        )
        return weights

    def build_weights_series(
        self,
        gds: pd.DataFrame,
        eligible_universe: Optional[dict[pd.Timestamp, list[str]]] = None,
    ) -> pd.DataFrame:
        """
        Constrói pesos para todas as datas de rebalanceamento.

        Returns
        -------
        pd.DataFrame com pesos por data e ativo.
        index=rebalance_dates, cols=tickers.
        """
        all_weights = {}

        for date in gds.index:
            gds_t = gds.loc[date]
            eligible = None
            if eligible_universe is not None:
                eligible = eligible_universe.get(date)

            w = self.build_weights(gds_t, eligible)
            if not w.empty:
                all_weights[date] = w

        if not all_weights:
            raise RuntimeError("Nenhuma data gerou pesos válidos.")

        weights_df = pd.DataFrame(all_weights).T.fillna(0.0)
        weights_df.index.name = "Date"

        logger.info(
            f"Pesos construídos: {weights_df.shape[0]} datas × {weights_df.shape[1]} ativos | "
            f"Pesos não-zero por data (média): "
            f"{(weights_df != 0).sum(axis=1).mean():.1f}"
        )
        return weights_df

    # ──────────────────────────────────────────────────────────────
    # UTILITÁRIOS
    # ──────────────────────────────────────────────────────────────
    def _apply_weight_cap(self, weights: pd.Series) -> pd.Series:
        """
        Cap de peso máximo por ativo para evitar concentração.
        Excesso é redistribuído uniformemente nos outros ativos da mesma perna.

        Se o número de ativos é tão pequeno que peso igual já excede o cap
        (ex: 4 ativos com cap=15% → peso igual=25%), o cap não é aplicado
        pois forçaria uma restrição infactível. Isso é logado como aviso.
        """
        cap = self.max_weight

        def _cap_leg(leg_weights: pd.Series) -> pd.Series:
            """Aplica cap em uma perna (positiva ou negativa)."""
            n = len(leg_weights)
            if n == 0:
                return leg_weights
            # Se peso igual já excede o cap, não há como redistribuir
            # (restrição infactível) — mantém igual e loga aviso
            if 1.0 / n > cap:
                logger.debug(
                    f"Cap de {cap:.0%} infactível com {n} ativos "
                    f"(peso igual = {1/n:.2%}). Usando pesos iguais."
                )
                return leg_weights  # já normalizado
            # Redistribuição iterativa
            max_iters = 20
            for _ in range(max_iters):
                over = leg_weights > cap
                if not over.any():
                    break
                excess = (leg_weights[over] - cap).sum()
                leg_weights[over] = cap
                under = leg_weights[~over]
                if len(under) > 0:
                    leg_weights[~over] += excess / len(under)
            return leg_weights

        # Long leg
        long_mask = weights > 0
        if long_mask.sum() > 0:
            weights[long_mask] = _cap_leg(weights[long_mask].copy())

        # Short leg (trabalhar com positivos)
        short_mask = weights < 0
        if short_mask.sum() > 0:
            short_pos = (-weights[short_mask]).copy()
            short_pos = _cap_leg(short_pos)
            weights[short_mask] = -short_pos

        return weights

    def _renormalize(self, weights: pd.Series) -> pd.Series:
        """
        Renormaliza após o cap para garantir que:
          Σ w_long = +1.0
          Σ w_short = -long_short_ratio
        """
        long_mask  = weights > 0
        short_mask = weights < 0

        if long_mask.sum() > 0:
            long_sum = weights[long_mask].sum()
            if long_sum > 1e-9:
                weights[long_mask] /= long_sum  # soma a 1

        if short_mask.sum() > 0:
            short_sum = weights[short_mask].abs().sum()
            if short_sum > 1e-9:
                weights[short_mask] *= (self.long_short_ratio / short_sum)

        return weights

    def get_turnover(
        self,
        weights_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Calcula turnover entre rebalanceamentos consecutivos.

        Turnover(t) = Σ_i |w_i(t) - w_i(t-1)| / 2

        Turnover = 1.0 significa que todo o portfólio foi renovado.
        Usado para estimar custos de transação.
        """
        diff = weights_df.diff().abs().sum(axis=1) / 2.0
        diff.iloc[0] = weights_df.iloc[0].abs().sum() / 2.0  # primeiro rebal = comprar tudo
        return diff
