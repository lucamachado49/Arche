"""
src/signal/generator.py
───────────────────────
Gerador de sinais GDS ponto-no-tempo para o loop de backtest.

Orquestra o pipeline completo:
  preços → retornos → grafo → curvatura → κ_i(t) → GDS(t) → sinal

O sinal gerado aqui alimenta o PortfolioConstructor em src/portfolio/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

from src.data.cleaner import DataCleaner
from src.data.universe import UniverseBuilder
from src.graph.builder import GraphBuilder
from src.curvature.curvature import OllivierRicciCurvature
from src.curvature.gds import GDSComputer
from src.data.config import get


class SignalGenerator:
    """
    Gera sinais GDS para todas as datas de rebalanceamento.

    Este é o módulo de orquestração central do projeto.
    Pode ser usado em modo completo (constrói grafo a cada data)
    ou modo de cache (carrega curvaturas pré-computadas).

    Uso típico
    ----------
    generator = SignalGenerator()
    kappa_ts, gds_ts = generator.run(
        prices=prices,
        volumes=volumes,
        rebalance_dates=dates,
    )
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cfg         = get("graph")
        self.window_days = self.cfg["correlation_window_days"]
        self.cache_dir   = cache_dir

        self.cleaner      = DataCleaner()
        self.graph_builder = GraphBuilder()
        self.orc          = OllivierRicciCurvature()
        self.gds_computer  = GDSComputer()

    # ──────────────────────────────────────────────────────────────
    # INTERFACE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def run(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
        n_jobs: int = 1,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pipeline completo de geração de sinal.

        Parameters
        ----------
        prices          : preços ajustados (index=Date, cols=tickers)
        volumes         : volumes em R$ (opcional, para filtro de liquidez)
        rebalance_dates : datas de rebalanceamento. Se None, usa todas as sextas.
        n_jobs          : paralelismo no cálculo da curvatura
        use_cache       : salvar/carregar κ_i de disco para evitar recomputação

        Returns
        -------
        (kappa_ts, gds_ts): DataFrames de κ_i e GDS por data/ativo
        """
        # 1. Calcular retornos limpos
        logger.info("Passo 1/4: Calculando retornos...")
        returns = self.cleaner.prices_to_log_returns(prices)

        # 2. Determinar datas de rebalanceamento
        if rebalance_dates is None:
            rebalance_dates = self._get_weekly_dates(returns.index)
        logger.info(f"Passo 2/4: {len(rebalance_dates)} datas de rebalanceamento")

        # 3. Construir universo ponto-no-tempo
        universe_builder = UniverseBuilder(prices, volumes)

        # 4. Tentar carregar cache
        if use_cache and self.cache_dir is not None:
            cached = self._try_load_cache()
            if cached is not None:
                kappa_ts, gds_ts = cached
                logger.info("Cache de curvatura carregado.")
                return kappa_ts, gds_ts

        # 5. Calcular curvatura em cada data
        logger.info("Passo 3/4: Calculando curvatura de Ollivier-Ricci...")
        kappa_records = self._compute_kappa_loop(
            returns=returns,
            rebalance_dates=rebalance_dates,
            universe_builder=universe_builder,
            n_jobs=n_jobs,
        )

        if not kappa_records:
            raise RuntimeError("Nenhuma data produziu curvatura válida.")

        kappa_ts = pd.DataFrame(kappa_records).T.sort_index()
        kappa_ts.index.name = "Date"

        # 6. Calcular GDS
        logger.info("Passo 4/4: Calculando GDS...")
        gds_ts = self.gds_computer.compute_gds_series(kappa_ts)

        # 7. Salvar cache
        if use_cache and self.cache_dir is not None:
            self._save_cache(kappa_ts, gds_ts)

        logger.info(
            f"✓ Pipeline completo | κ: {kappa_ts.shape} | GDS: {gds_ts.shape}"
        )
        return kappa_ts, gds_ts

    # ──────────────────────────────────────────────────────────────
    # LOOP DE CURVATURA (look-ahead safe)
    # ──────────────────────────────────────────────────────────────
    def _compute_kappa_loop(
        self,
        returns: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        universe_builder: UniverseBuilder,
        n_jobs: int = 1,
    ) -> dict[pd.Timestamp, pd.Series]:
        """
        Loop temporal: para cada data t, constrói o grafo com dados até t
        e calcula a curvatura nodal κ_i(t).

        INVARIANTE DE LOOK-AHEAD:
        - window = retornos até t (inclusive)
        - eligible = ativos com liquidez até t-1
        - κ_i(t) usa apenas informação disponível em t ✓
        """
        kappa_records = {}
        n_total = len(rebalance_dates)

        for i, date in enumerate(rebalance_dates):
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{n_total}: {date.date()}")

            # Janela de retornos até t (look-ahead safe)
            window = self.cleaner.get_returns_window(
                returns, date, self.window_days
            )

            if len(window) < max(20, self.window_days // 3):
                logger.debug(f"  {date.date()}: janela insuficiente ({len(window)}d)")
                continue

            # Universo elegível em t (usa dados até t)
            eligible = universe_builder.get_eligible(date)
            if len(eligible) < 5:
                logger.warning(f"  {date.date()}: menos de 5 ativos elegíveis")
                continue

            try:
                # Construir grafo
                G = self.graph_builder.build(window, eligible)

                # Calcular curvatura
                G = self.orc.compute(G, n_jobs=n_jobs)

                # Curvatura nodal
                nodal = self.gds_computer.get_nodal_kappa(G)
                kappa_records[date] = nodal

            except Exception as e:
                logger.error(f"  {date.date()}: erro — {e}")

        return kappa_records

    # ──────────────────────────────────────────────────────────────
    # UTILITÁRIOS
    # ──────────────────────────────────────────────────────────────
    def _get_weekly_dates(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Retorna sextas-feiras úteis dentro do index.
        Estratégia de rebalanceamento semanal padrão.
        """
        fridays = index[index.dayofweek == 4]  # 4 = sexta
        # Garantir gap mínimo de dados antes de começar
        min_history = get("universe", "min_history_days", 252)
        if len(index) > min_history:
            start = index[min_history]
            fridays = fridays[fridays >= start]
        return fridays

    def _save_cache(
        self,
        kappa_ts: pd.DataFrame,
        gds_ts: pd.DataFrame,
    ) -> None:
        """Salva κ_i e GDS em parquet para evitar recomputação."""
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        kappa_ts.to_parquet(self.cache_dir / "kappa_ts.parquet")
        gds_ts.to_parquet(self.cache_dir / "gds_ts.parquet")
        logger.info(f"Cache salvo em: {self.cache_dir}")

    def _try_load_cache(self) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
        """Tenta carregar κ_i e GDS do cache em disco."""
        if self.cache_dir is None:
            return None
        kappa_path = self.cache_dir / "kappa_ts.parquet"
        gds_path   = self.cache_dir / "gds_ts.parquet"
        if kappa_path.exists() and gds_path.exists():
            try:
                return (
                    pd.read_parquet(kappa_path),
                    pd.read_parquet(gds_path),
                )
            except Exception as e:
                logger.warning(f"Falha ao carregar cache: {e}")
        return None
