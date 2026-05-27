"""
cleaner.py
──────────
Responsabilidade única: transformar preços brutos em retornos
logarítmicos limpos, prontos para o grafo.

REGRAS CRÍTICAS DE LOOK-AHEAD SAFETY
──────────────────────────────────────
1. Nenhuma operação usa informação do dia t para calcular
   algo que vai ser usado como "disponível em t-1".
2. Retornos log são calculados como ln(P_t / P_{t-1}) —
   o retorno do dia t usa o preço de fechamento de t e t-1.
   Isso é informação disponível no final do dia t. ✓
3. Preenchimento de NaN: usa apenas FORWARD FILL (ffill) —
   nunca backward fill (bfill) que vazaria dados futuros. ✓
4. Outlier winsorization usa percentis calculados apenas no
   passado (expanding, não rolling com dados futuros). ✓
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import get


class DataCleaner:
    """
    Pipeline de limpeza de dados brutos.

    Passos executados na ordem:
    1. Forward fill de NaN (máximo 5 dias — suspensão de negociação)
    2. Remoção de ativos com histórico insuficiente
    3. Cálculo de retornos logarítmicos
    4. Detecção e winsorização de outliers extremos
    5. Alinhamento de datas entre múltiplos DataFrames

    Uso típico
    ----------
    cleaner = DataCleaner()
    returns = cleaner.prices_to_log_returns(prices_df)
    """

    # Constantes de sanidade
    MAX_FFILL_DAYS      = 5       # máx dias de forward fill (suspensão de negociação)
    MIN_RETURN_PERIODS  = 252     # mínimo de retornos não-NaN para manter ativo
    OUTLIER_THRESHOLD   = 0.50    # retorno > 50% ou < -50% em 1 dia é suspeito
    WINSOR_CLIP         = 0.40    # winsoriza em ±40% (ex: abertura de capital)

    def __init__(self) -> None:
        self.cfg = get("universe")

    # ──────────────────────────────────────────────────────────────
    # PIPELINE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def prices_to_log_returns(
        self,
        prices: pd.DataFrame,
        winsorize: bool = True,
    ) -> pd.DataFrame:
        """
        Converte preços em retornos logarítmicos limpos.

        Parameters
        ----------
        prices    : DataFrame de preços (index=DatetimeIndex, cols=tickers)
        winsorize : se True, aplica winsorização de outliers extremos

        Returns
        -------
        pd.DataFrame de retornos log, mesmo shape menos a primeira linha.
        """
        logger.info(f"Iniciando pipeline de limpeza: {prices.shape}")

        # 1. Forward fill (máx 5 dias — suspensão temporária de negociação)
        prices_filled = self._forward_fill(prices)

        # 2. Remover ativos com histórico insuficiente
        prices_filtered = self._filter_min_history(prices_filled)

        # 3. Calcular retornos logarítmicos
        returns = self._compute_log_returns(prices_filtered)

        # 4. Winsorizar outliers extremos
        if winsorize:
            returns = self._winsorize_returns(returns)

        # 5. Relatório final
        self._log_cleaning_report(prices, returns)

        return returns

    def align_with_factors(
        self,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Alinha retornos e fatores NEFIN no mesmo DatetimeIndex.

        Retorna apenas as datas onde AMBOS têm dados disponíveis.
        Isso é crítico: não queremos preencher dias sem fator com NaN
        — isso distorceria a regressão Fama-French.

        Parameters
        ----------
        returns : DataFrame de retornos B3
        factors : DataFrame de fatores NEFIN

        Returns
        -------
        Tupla (returns_aligned, factors_aligned) com mesmo index.
        """
        common_dates = returns.index.intersection(factors.index)

        if len(common_dates) < 252:
            raise ValueError(
                f"Apenas {len(common_dates)} datas em comum entre retornos e fatores. "
                f"Verifique os períodos cobertos por cada dataset."
            )

        n_dropped_returns = len(returns) - len(common_dates)
        n_dropped_factors = len(factors) - len(common_dates)

        if n_dropped_returns > 0:
            logger.warning(
                f"Retornos: {n_dropped_returns} datas sem fatores NEFIN correspondentes — removidas."
            )
        if n_dropped_factors > 0:
            logger.warning(
                f"Fatores: {n_dropped_factors} datas sem retornos correspondentes — removidas."
            )

        logger.info(
            f"Alinhamento OK: {len(common_dates)} datas em comum "
            f"({common_dates[0].date()} → {common_dates[-1].date()})"
        )

        return returns.loc[common_dates], factors.loc[common_dates]

    # ──────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS
    # ──────────────────────────────────────────────────────────────
    def _forward_fill(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica forward fill para tratar suspensões temporárias.

        SEGURANÇA DE LOOK-AHEAD:
        ffill propaga o último preço CONHECIDO para frente.
        Isso simula "último preço disponível" — correto operacionalmente.
        Nunca usamos bfill (backward fill) pois preencheria com preços futuros.
        """
        n_nan_before = prices.isna().sum().sum()
        filled = prices.ffill(limit=self.MAX_FFILL_DAYS)
        n_nan_after = filled.isna().sum().sum()
        filled_count = n_nan_before - n_nan_after

        if filled_count > 0:
            logger.debug(
                f"Forward fill: {filled_count} NaN preenchidos "
                f"(máx {self.MAX_FFILL_DAYS} dias consecutivos)"
            )

        return filled

    def _filter_min_history(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Remove ativos com menos que MIN_RETURN_PERIODS observações válidas.

        Ativos com histórico curto demais distorcem as correlações e
        a curvatura de Ricci — melhor excluir.
        """
        valid_obs = prices.notna().sum()
        mask_keep = valid_obs >= self.MIN_RETURN_PERIODS
        n_removed = (~mask_keep).sum()

        if n_removed > 0:
            removed = prices.columns[~mask_keep].tolist()
            logger.info(
                f"Removidos {n_removed} ativos com < {self.MIN_RETURN_PERIODS} "
                f"obs válidas: {removed[:10]}{'...' if len(removed)>10 else ''}"
            )

        return prices.loc[:, mask_keep]

    def _compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula retornos logarítmicos: r_t = ln(P_t / P_{t-1}).

        Por que log e não retorno simples?
        - Aditivo no tempo: r(0→2) = r(0→1) + r(1→2)
        - Aproxima retorno simples para |r| < 10%
        - Mais robusto a outliers extremos
        - Pressuposto padrão em modelos financeiros

        LOOK-AHEAD: retorno do dia t usa apenas P_t e P_{t-1}. ✓
        """
        returns = np.log(prices / prices.shift(1))

        # Primeira linha é sempre NaN (sem preço anterior) — remover
        returns = returns.iloc[1:]

        n_nan = returns.isna().sum().sum()
        total = returns.size
        logger.debug(
            f"Retornos log calculados: shape={returns.shape}, "
            f"NaN restantes={n_nan} ({n_nan/total:.1%})"
        )

        return returns

    def _winsorize_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Winsoriza retornos extremos usando thresholds fixos.

        Diferente de winsorização por percentil calculada in-sample,
        usamos thresholds fixos (±40%) para evitar look-ahead:
        - Percentis calculados no período inteiro vazariam futuro
        - Thresholds fixos são regras operacionais, não estimadas

        Retornos > ±50% são SINALIZADOS como suspeitos.
        Retornos > ±40% são CLIPADOS.

        Atenção: IPOs, circuit breakers e erros de dado frequentemente
        geram outliers que destroem correlações e curvatura.
        """
        # Sinalizar antes de clivar
        extreme_mask = returns.abs() > self.OUTLIER_THRESHOLD
        n_extreme = extreme_mask.sum().sum()

        if n_extreme > 0:
            extreme_by_ticker = extreme_mask.sum().sort_values(ascending=False)
            top_extreme = extreme_by_ticker[extreme_by_ticker > 0].head(10)
            logger.warning(
                f"{n_extreme} retornos extremos (|r| > {self.OUTLIER_THRESHOLD:.0%}) detectados.\n"
                f"Top ativos afetados:\n{top_extreme.to_string()}"
            )

        # Clivar em ±WINSOR_CLIP
        returns_winsorized = returns.clip(
            lower=-self.WINSOR_CLIP,
            upper=self.WINSOR_CLIP,
        )

        n_clipped = (returns_winsorized != returns).sum().sum()
        if n_clipped > 0:
            logger.info(f"Winsorização: {n_clipped} retornos clipados em ±{self.WINSOR_CLIP:.0%}")

        return returns_winsorized

    def _log_cleaning_report(
        self,
        prices_original: pd.DataFrame,
        returns_clean: pd.DataFrame,
    ) -> None:
        """Loga relatório final do processo de limpeza."""
        fill_rate = returns_clean.notna().mean().mean()
        logger.info(
            f"Pipeline de limpeza concluído:\n"
            f"  Ativos originais  : {prices_original.shape[1]}\n"
            f"  Ativos finais     : {returns_clean.shape[1]}\n"
            f"  Período           : {returns_clean.index[0].date()} → {returns_clean.index[-1].date()}\n"
            f"  Obs. por ativo    : {len(returns_clean)} dias\n"
            f"  Fill rate médio   : {fill_rate:.1%}\n"
            f"  Retorno médio     : {returns_clean.mean().mean():.4%}/dia\n"
            f"  Vol. média        : {returns_clean.std().mean():.4%}/dia"
        )

    # ──────────────────────────────────────────────────────────────
    # MÉTODOS AUXILIARES PÚBLICOS
    # ──────────────────────────────────────────────────────────────
    def get_returns_window(
        self,
        returns: pd.DataFrame,
        end_date: pd.Timestamp,
        window_days: int,
    ) -> pd.DataFrame:
        """
        Extrai uma janela de retornos terminando em end_date.

        CRÍTICO PARA LOOK-AHEAD SAFETY:
        Esta função é chamada no loop do backtest para garantir que
        cada cálculo de grafo/curvatura só vê dados até end_date.

        Parameters
        ----------
        returns   : DataFrame completo de retornos
        end_date  : última data da janela (inclusive)
        window_days : tamanho da janela em dias úteis

        Returns
        -------
        Janela de retornos com no máximo window_days linhas.
        """
        # Apenas datas até end_date (exclusive futuro)
        available = returns.loc[:end_date]

        if len(available) < window_days:
            logger.debug(
                f"Janela em {end_date.date()}: apenas {len(available)} dias "
                f"disponíveis (pedido: {window_days})"
            )
            return available

        return available.iloc[-window_days:]

    def compute_rolling_stats(
        self,
        returns: pd.DataFrame,
        window: int = 60,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcula média e desvio-padrão rolling de retornos.
        Usado para normalizar o GDS por ativo.

        Returns
        -------
        (rolling_mean, rolling_std) — ambos com mesmo shape de returns.
        """
        rolling_mean = returns.rolling(window=window, min_periods=window // 2).mean()
        rolling_std  = returns.rolling(window=window, min_periods=window // 2).std()

        # std = 0 causaria divisão por zero no GDS — substituir por NaN
        rolling_std = rolling_std.replace(0, np.nan)

        return rolling_mean, rolling_std
