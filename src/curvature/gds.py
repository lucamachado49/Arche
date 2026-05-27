"""
src/curvature/gds.py
────────────────────
Geometric Divergence Score (GDS) — coração da hipótese do Projeto Arche.

DEFINIÇÃO FORMAL
─────────────────
O GDS de um ativo i na data t é definido como:

    GDS_i(t) = Δκ_i(t) / σ_i

onde:
  Δκ_i(t) = κ_i(t) - κ_i(t-1)   (variação da curvatura nodal)
  κ_i(t)  = média das curvaturas das arestas incidentes em i:
             κ_i(t) = (1/|N(i)|) Σ_{j∈N(i)} κ(i,j;t)
  σ_i     = desvio padrão histórico de κ_i (janela configurável)

HIPÓTESE TESTÁVEL
──────────────────
    E[r_i(t+1:t+h) | GDS_i(t) < -z·σ_i] > E[r_i(t+1:t+h)] + α

Em palavras: ativo com GDS anormalmente NEGATIVO (curvatura colapsou)
está geometricamente mal posicionado → espera-se reversão → alpha positivo.

LOOK-AHEAD SAFETY
──────────────────
O GDS em t usa κ calculado sobre janela de retornos até t.
O sinal GDS(t) é usado para prever retorno de t+1 em diante.
Nunca há cruzamento temporal. ✓
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

from src.data.config import get
from .curvature import OllivierRicciCurvature


class GDSComputer:
    """
    Calcula o Geometric Divergence Score (GDS) para cada ativo em cada data.

    Pipeline:
    1. Para cada data de rebalanceamento t:
       a. Construir grafo com janela de retornos até t  (GraphBuilder)
       b. Calcular curvatura κ(i,j) para cada aresta   (OllivierRicciCurvature)
       c. Calcular curvatura nodal κ_i = média das arestas de i
    2. Calcular série temporal de κ_i(t) para cada ativo
    3. Calcular GDS = Δκ_i / σ_i usando janela histórica

    Uso típico
    ----------
    computer = GDSComputer()

    # Loop de backtest (look-ahead safe):
    kappa_history = {}  # acumular ao longo do tempo
    for date in rebalance_dates:
        window = cleaner.get_returns_window(returns, date, window_days=60)
        G = graph_builder.build(window, eligible_tickers)
        G = orc.compute(G)
        nodal_kappa = computer.get_nodal_kappa(G)
        kappa_history[date] = nodal_kappa
        gds = computer.compute_gds(kappa_history)
    """

    def __init__(self) -> None:
        cfg = get("curvature")
        self.kappa_history_days = cfg.get("kappa_history_days", 252)
        self.gds_threshold_sigma = cfg.get("gds_threshold_sigma", 2.0)
        self.orc = OllivierRicciCurvature()

    # ──────────────────────────────────────────────────────────────
    # INTERFACE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def get_nodal_kappa(self, G: nx.Graph) -> pd.Series:
        """
        Calcula a curvatura nodal κ_i para cada nó do grafo.

        κ_i = média aritmética das curvaturas das arestas incidentes em i.

        Por que média e não outra agregação?
        - Reflete a curvatura "típica" do ambiente local do ativo
        - Robusta a outliers de uma única aresta
        - Consistente com Sandhu et al. (2016)

        Parameters
        ----------
        G : grafo com atributo "kappa" em cada aresta

        Returns
        -------
        pd.Series: index=tickers, values=κ_i (NaN se ativo sem arestas com κ)
        """
        nodal = {}
        for node in G.nodes():
            edge_kappas = []
            for neighbor in G.neighbors(node):
                k = G[node][neighbor].get("kappa", None)
                if k is not None and np.isfinite(k):
                    edge_kappas.append(k)

            if edge_kappas:
                nodal[node] = float(np.mean(edge_kappas))
            else:
                nodal[node] = np.nan

        return pd.Series(nodal, name="kappa_nodal")

    def compute_gds_series(
        self,
        kappa_ts: pd.DataFrame,
        min_history: int = 21,
    ) -> pd.DataFrame:
        """
        Calcula a série temporal de GDS para cada ativo.

        GDS_i(t) = Δκ_i(t) / σ_i(t)

        onde σ_i(t) é calculado sobre a janela histórica de κ_i
        até t (expansível, nunca usa dados futuros).

        Parameters
        ----------
        kappa_ts   : DataFrame com κ_i(t) por data e ativo
                     index=DatetimeIndex, columns=tickers
        min_history: mínimo de observações para calcular σ_i

        Returns
        -------
        pd.DataFrame com GDS(t) por data e ativo.
        Mesmas dimensões de kappa_ts menos a primeira linha.
        """
        if kappa_ts.empty or len(kappa_ts) < 2:
            logger.warning("Série de κ muito curta para calcular GDS.")
            return pd.DataFrame()

        # Δκ_i(t) = κ_i(t) - κ_i(t-1)
        delta_kappa = kappa_ts.diff().iloc[1:]

        # σ_i(t): desvio-padrão rolling expandível (expanding window)
        # LOOK-AHEAD SAFE: expanding nunca usa dados futuros ✓
        kappa_std = kappa_ts.expanding(min_periods=min_history).std().iloc[1:]

        # GDS = Δκ / σ  (z-score da variação de curvatura)
        gds = delta_kappa / kappa_std.replace(0, np.nan)

        # Clamp para evitar z-scores absurdos por instabilidade numérica
        gds = gds.clip(-10, 10)

        logger.debug(
            f"GDS calculado: {gds.shape} | "
            f"média={gds.mean().mean():.4f}, "
            f"std={gds.std().mean():.4f}"
        )
        return gds

    def get_signal_mask(
        self,
        gds: pd.DataFrame,
        threshold_z: Optional[float] = None,
        direction: str = "negative",
    ) -> pd.DataFrame:
        """
        Gera máscara booleana de ativos com GDS além do threshold.

        A hipótese central usa:
          - direction="negative": GDS < -z  (curvatura colapsou → long)
          - direction="positive": GDS > +z  (curvatura expandiu → short)

        Parameters
        ----------
        gds         : DataFrame de GDS(t)
        threshold_z : z-score limiar (usa settings.yaml se None)
        direction   : "negative" | "positive" | "both"

        Returns
        -------
        pd.DataFrame booleano: True = ativo com sinal na data t.
        """
        z = threshold_z or self.gds_threshold_sigma

        if direction == "negative":
            return gds < -z
        elif direction == "positive":
            return gds > z
        elif direction == "both":
            return gds.abs() > z
        else:
            raise ValueError(f"direction deve ser 'negative', 'positive' ou 'both'. Got: {direction}")

    def full_pipeline(
        self,
        rebalance_dates: pd.DatetimeIndex,
        returns: pd.DataFrame,
        graph_builder,
        eligible_universe: dict[pd.Timestamp, list[str]],
        window_days: int = 60,
        n_jobs: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pipeline completo: datas → κ_i(t) → GDS(t).

        LOOK-AHEAD INVARIANTE:
        Para cada data t, usa apenas retornos até t para construir
        o grafo e calcular a curvatura. ✓

        Parameters
        ----------
        rebalance_dates   : datas de rebalanceamento (DatetimeIndex)
        returns           : DataFrame de retornos log (index=Date, cols=tickers)
        graph_builder     : instância de GraphBuilder
        eligible_universe : dict {date: lista de tickers elegíveis}
        window_days       : tamanho da janela de retornos para o grafo
        n_jobs            : paralelismo para ORC

        Returns
        -------
        (kappa_ts, gds_ts): DataFrames de κ_i e GDS por data/ativo
        """
        from src.data.cleaner import DataCleaner
        cleaner = DataCleaner()

        kappa_records = {}

        for date in rebalance_dates:
            logger.info(f"Processando {date.date()}...")

            # 1. Janela de retornos (look-ahead safe)
            window = cleaner.get_returns_window(returns, date, window_days)
            eligible = eligible_universe.get(date, list(returns.columns))

            if len(window) < 20:
                logger.warning(f"{date.date()}: janela muito curta ({len(window)} dias), pulando")
                continue

            try:
                # 2. Construir grafo
                G = graph_builder.build(window, eligible)

                # 3. Calcular curvatura
                G = self.orc.compute(G, n_jobs=n_jobs)

                # 4. Curvatura nodal
                nodal_kappa = self.get_nodal_kappa(G)
                kappa_records[date] = nodal_kappa

            except Exception as e:
                logger.error(f"{date.date()}: erro no pipeline de curvatura — {e}")
                continue

        if not kappa_records:
            raise RuntimeError("Nenhuma data processada com sucesso no pipeline de curvatura.")

        # Montar DataFrame de κ_i(t)
        kappa_ts = pd.DataFrame(kappa_records).T.sort_index()
        kappa_ts.index.name = "Date"

        # Calcular GDS
        gds_ts = self.compute_gds_series(kappa_ts)

        logger.info(
            f"Pipeline concluído: κ shape={kappa_ts.shape}, GDS shape={gds_ts.shape}"
        )
        return kappa_ts, gds_ts

    # ──────────────────────────────────────────────────────────────
    # DIAGNÓSTICO / SANIDADE
    # ──────────────────────────────────────────────────────────────
    def sanity_check_kappa(
        self,
        kappa_ts: pd.DataFrame,
        sector_map: Optional[dict[str, str]] = None,
    ) -> dict[str, bool]:
        """
        Gate de saída do Sprint 3: verifica se κ_i de bancos
        é maior que κ_i de ativos idiossincráticos.

        Bancos têm curvatura maior porque são altamente inter-conectados
        (formam um cluster denso). Ativos idiossincráticos são "pontes"
        no grafo — curvatura negativa é esperada.
        """
        if sector_map is None:
            sector_map = {
                "ITUB4": "banco", "BBDC4": "banco", "BBAS3": "banco",
                "SANB11": "banco", "BPAC11": "banco",
                "PRIO3": "idioss", "IRBR3": "idioss",
                "LWSA3": "idioss", "AMER3": "idioss",
            }

        checks = {}
        available = set(kappa_ts.columns)

        bancos    = [t for t in sector_map if sector_map[t] == "banco" and t in available]
        idioss    = [t for t in sector_map if sector_map[t] == "idioss" and t in available]

        if not bancos or not idioss:
            logger.warning("Tickers de referência não disponíveis para sanity check.")
            checks["bancos_vs_idioss"] = None
            return checks

        kappa_bancos = kappa_ts[bancos].mean().mean()
        kappa_idioss = kappa_ts[idioss].mean().mean()

        checks["bancos_vs_idioss"] = kappa_bancos > kappa_idioss
        checks["kappa_bancos_mean"] = round(kappa_bancos, 4)
        checks["kappa_idioss_mean"] = round(kappa_idioss, 4)

        status = "✓" if checks["bancos_vs_idioss"] else "✗"
        logger.info(
            f"Sanity check κ: {status} "
            f"κ_bancos={kappa_bancos:.4f} vs κ_idioss={kappa_idioss:.4f}"
        )
        return checks
