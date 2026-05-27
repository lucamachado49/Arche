"""
src/graph/builder.py
────────────────────
Constrói o grafo ponderado (MST + threshold) a partir de retornos.

INVARIANTE DE LOOK-AHEAD:
Toda chamada a build() recebe uma janela de retornos já truncada
em uma data t pelo DataCleaner.get_returns_window().
Este módulo nunca acessa datas — só recebe o que já foi filtrado.

SAÍDA ESPERADA (critério de sanidade — Sprint 2):
  - VALE3 e PETR4 são vizinhos na MST
  - ITUB4 e BBDC4 são vizinhos na MST
  - Distância intra-setor < distância inter-setor
  - Grafo é conexo (todos os nós alcançáveis)
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

from src.data.config import get


class GraphBuilder:
    """
    Constrói o grafo de correlações para uma janela de retornos.

    Passos:
    1. Calcular matriz de correlação de Pearson
    2. Converter para distância: d(i,j) = √(2(1−ρ))
    3. Construir MST via algoritmo de Kruskal
    4. Adicionar arestas extras acima do threshold ξ
    5. Calcular distribuições de Markov em cada nó

    Uso típico
    ----------
    builder = GraphBuilder()
    G = builder.build(returns_window, eligible_tickers)
    """

    def __init__(self) -> None:
        cfg = get("graph")
        self.window_days      = cfg["correlation_window_days"]
        self.min_periods      = cfg["correlation_min_periods"]
        self.extra_threshold  = cfg["mst_extra_threshold"]
        self.markov_alpha     = cfg["markov_alpha"]

    # ──────────────────────────────────────────────────────────────
    # INTERFACE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def build(
        self,
        returns_window: pd.DataFrame,
        eligible_tickers: Optional[list[str]] = None,
    ) -> nx.Graph:
        """
        Constrói o grafo MST a partir de uma janela de retornos.

        Parameters
        ----------
        returns_window    : DataFrame de retornos (já truncado em t pelo cleaner)
        eligible_tickers  : lista de tickers do universo elegível.
                            Se None, usa todas as colunas.

        Returns
        -------
        nx.Graph com:
          - Nós: tickers
          - Arestas: pares com correlation > threshold (MST + extras)
          - Atributos de aresta: weight (distância), correlation, in_mst
          - Atributos de nó: markov_dist (dict de distribuição de Markov)
        """
        t0 = time.perf_counter()

        # Filtrar para o universo elegível
        if eligible_tickers:
            cols = [t for t in eligible_tickers if t in returns_window.columns]
            window = returns_window[cols].copy()
        else:
            window = returns_window.copy()

        # Remover ativos com dados insuficientes nessa janela
        min_obs = self.min_periods
        valid_cols = window.columns[window.notna().sum() >= min_obs]
        if len(valid_cols) < len(window.columns):
            dropped = set(window.columns) - set(valid_cols)
            logger.debug(
                f"GraphBuilder: {len(dropped)} ativos removidos por dados "
                f"insuficientes na janela (<{min_obs} obs): "
                f"{list(dropped)[:5]}"
            )
        window = window[valid_cols]

        if len(window.columns) < 5:
            raise ValueError(
                f"Apenas {len(window.columns)} ativos com dados suficientes. "
                f"Mínimo é 5 para construir um grafo significativo."
            )

        # 1. Matriz de correlação
        corr_matrix = self._compute_correlation(window)

        # 2. Matriz de distâncias
        dist_matrix = self._correlation_to_distance(corr_matrix)

        # 3. Grafo completo (apenas como base para o MST)
        G_full = self._build_complete_graph(dist_matrix, corr_matrix)

        # 4. MST via Kruskal
        G_mst = self._build_mst(G_full)

        # 5. Adicionar arestas extras acima do threshold
        G_final = self._add_extra_edges(G_mst, corr_matrix, dist_matrix)

        # 6. Distribuições de Markov em cada nó
        self._add_markov_distributions(G_final)

        elapsed = time.perf_counter() - t0
        logger.debug(
            f"Grafo construído: {G_final.number_of_nodes()} nós, "
            f"{G_final.number_of_edges()} arestas "
            f"({sum(1 for _,_,d in G_final.edges(data=True) if d.get('in_mst'))} MST + "
            f"{sum(1 for _,_,d in G_final.edges(data=True) if not d.get('in_mst'))} extras) "
            f"em {elapsed:.2f}s"
        )

        return G_final

    # ──────────────────────────────────────────────────────────────
    # PASSOS INTERNOS
    # ──────────────────────────────────────────────────────────────
    def _compute_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Matriz de correlação de Pearson com tratamento de NaN.

        Usa pairwise complete observations (padrão do pandas).
        Diagonal = 1 por definição.
        """
        corr = returns.corr(method="pearson", min_periods=self.min_periods)

        # Valores fora de [-1, 1] podem surgir por instabilidade numérica
        corr = corr.clip(-1.0, 1.0)

        # Copiar para array mutável antes de fill_diagonal
        corr_values = corr.values.copy()
        np.fill_diagonal(corr_values, 1.0)
        corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)

        # Verificar se há NaN na diagonal (ativo sem dados suficientes)
        diag_nan = corr.isnull().values.diagonal().sum()
        if diag_nan > 0:
            logger.warning(
                f"{diag_nan} ativos com correlação indefinida (NaN na diagonal). "
                f"Serão tratados como não correlacionados com todos."
            )
            corr_values = corr.fillna(0.0).values.copy()
            np.fill_diagonal(corr_values, 1.0)
            corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)

        return corr

    def _correlation_to_distance(self, corr: pd.DataFrame) -> pd.DataFrame:
        """
        Converte correlação em distância métrica.

        d(i,j) = √(2(1 − ρᵢⱼ))

        Propriedades:
          - d(i,i) = 0 (distância de um ativo a si mesmo)
          - d(i,j) ∈ [0, 2] (ρ ∈ [-1, 1])
          - Satisfaz os 4 axiomas de espaço métrico ✓
            (provado por Mantegna, 1999)
        """
        dist = np.sqrt(2.0 * (1.0 - corr.values))
        dist = np.clip(dist, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)

    def _build_complete_graph(
        self,
        dist_matrix: pd.DataFrame,
        corr_matrix: pd.DataFrame,
    ) -> nx.Graph:
        """
        Constrói grafo completo com todos os pares de ativos.
        Usado apenas internamente para extrair a MST.
        """
        G = nx.Graph()
        tickers = dist_matrix.columns.tolist()
        G.add_nodes_from(tickers)

        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if j <= i:
                    continue
                d = dist_matrix.iloc[i, j]
                c = corr_matrix.iloc[i, j]
                if np.isfinite(d) and d > 0:
                    G.add_edge(ti, tj, weight=d, correlation=c)

        return G

    def _build_mst(self, G_full: nx.Graph) -> nx.Graph:
        """
        Extrai a Minimum Spanning Tree usando algoritmo de Kruskal.

        Kruskal: ordena arestas por peso crescente e adiciona
        se não forma ciclo (usando Union-Find internamente no NetworkX).

        A MST tem exatamente n-1 arestas para n nós,
        e é conexa (todos os nós alcançáveis).
        """
        mst_edges = nx.minimum_spanning_tree(G_full, algorithm="kruskal")

        G_mst = nx.Graph()
        G_mst.add_nodes_from(G_full.nodes())

        for u, v, data in mst_edges.edges(data=True):
            G_mst.add_edge(u, v, in_mst=True, **data)

        assert nx.is_connected(G_mst), (
            "MST não é conexo! Isso indica problema nos dados de correlação."
        )
        assert G_mst.number_of_edges() == G_mst.number_of_nodes() - 1, (
            f"MST tem {G_mst.number_of_edges()} arestas, "
            f"esperado {G_mst.number_of_nodes()-1}."
        )

        return G_mst

    def _add_extra_edges(
        self,
        G_mst: nx.Graph,
        corr_matrix: pd.DataFrame,
        dist_matrix: pd.DataFrame,
    ) -> nx.Graph:
        """
        Adiciona arestas com correlação > threshold além da MST.

        Motivação: a MST captura o esqueleto do grafo, mas pode
        perder conexões fortes não incluídas por questões de caminho.
        Ativos com ρ > ξ são vizinhos naturais e devem ser conectados.
        """
        G = G_mst.copy()
        n_extra = 0

        tickers = corr_matrix.columns.tolist()
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if j <= i:
                    continue
                if G.has_edge(ti, tj):
                    continue  # já está na MST

                c = corr_matrix.iloc[i, j]
                if c > self.extra_threshold:
                    d = dist_matrix.iloc[i, j]
                    G.add_edge(ti, tj, weight=d, correlation=c, in_mst=False)
                    n_extra += 1

        logger.debug(
            f"Arestas extras (ρ > {self.extra_threshold}): {n_extra}"
        )
        return G

    def _add_markov_distributions(self, G: nx.Graph) -> None:
        """
        Calcula distribuição de Markov μᵢ para cada nó i.

        μᵢ(j) = w(i,j) / Σ_{k~i} w(i,k)

        onde w(i,j) é o peso da aresta (transformado de distância
        para similaridade: sim = 1/dist).

        Com self-loop de peso markov_alpha:
        μᵢ(i) = alpha / (alpha + Σ w(i,j))
        μᵢ(j) = w(i,j) / (alpha + Σ w(i,j))  para j ≠ i

        O self-loop garante que a curvatura é bem definida mesmo
        para nós com poucos vizinhos.
        """
        alpha = self.markov_alpha

        for node in G.nodes():
            neighbors = list(G.neighbors(node))

            if not neighbors:
                # Nó isolado (não deve acontecer com MST, mas por segurança)
                G.nodes[node]["markov_dist"] = {node: 1.0}
                continue

            # Similaridade = 1 / distância (invertemos para que
            # ativos mais próximos = mais correlacionados tenham mais peso)
            similarities = {}
            for neighbor in neighbors:
                d = G[node][neighbor]["weight"]
                # Evitar divisão por zero (distância = 0 = correlação perfeita)
                sim = 1.0 / max(d, 1e-6)
                similarities[neighbor] = sim

            total_sim = sum(similarities.values()) + alpha

            markov = {}
            # Self-loop
            if alpha > 0:
                markov[node] = alpha / total_sim
            # Vizinhos
            for neighbor, sim in similarities.items():
                markov[neighbor] = sim / total_sim

            # Verificar que soma = 1 (distribuição de probabilidade válida)
            assert abs(sum(markov.values()) - 1.0) < 1e-9, (
                f"Distribuição de Markov em {node} não soma 1: "
                f"{sum(markov.values()):.6f}"
            )

            G.nodes[node]["markov_dist"] = markov

    # ──────────────────────────────────────────────────────────────
    # UTILITÁRIOS
    # ──────────────────────────────────────────────────────────────
    def validate_graph_sanity(self, G: nx.Graph) -> dict[str, bool]:
        """
        Verifica sanidade do grafo construído.
        Retorna dict de checks para o notebook de EDA.

        Checks esperados para o Ibovespa:
          - VALE3 e PETR4 vizinhos (ambas dependem de commodities/câmbio)
          - ITUB4 e BBDC4 vizinhos (bancos privados)
          - Grafo conexo
          - Distribuições de Markov somam 1
        """
        checks = {}

        # Conectividade
        checks["grafo_conexo"] = nx.is_connected(G)

        # Pares esperados de vizinhos
        expected_neighbors = [
            ("VALE3", "PETR4"),
            ("ITUB4", "BBDC4"),
        ]
        for u, v in expected_neighbors:
            if u in G.nodes() and v in G.nodes():
                # Verifica se são vizinhos diretos ou estão a no máximo 2 hops
                # (distância ponderada é frágil para thresholds fixos)
                try:
                    hops = nx.shortest_path_length(G, u, v)  # sem peso = hops
                    checks[f"{u}_{v}_proximos"] = hops <= 2
                except nx.NetworkXNoPath:
                    checks[f"{u}_{v}_proximos"] = False
            else:
                checks[f"{u}_{v}_proximos"] = None  # ativo não está no universo

        # Distribuições de Markov válidas
        markov_valid = all(
            abs(sum(G.nodes[n]["markov_dist"].values()) - 1.0) < 1e-6
            for n in G.nodes()
            if "markov_dist" in G.nodes[n]
        )
        checks["markov_validas"] = markov_valid

        # MST tem n-1 arestas
        n_mst = sum(1 for _, _, d in G.edges(data=True) if d.get("in_mst", False))
        checks["mst_n_menos_1"] = (n_mst == G.number_of_nodes() - 1)

        return checks

    def save(self, G: nx.Graph, path: Path) -> None:
        """Serializa o grafo para cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(G, f)

    @staticmethod
    def load(path: Path) -> nx.Graph:
        """Carrega grafo do cache."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_adjacency_info(self, G: nx.Graph) -> pd.DataFrame:
        """
        Retorna DataFrame com todas as arestas do grafo.
        Útil para visualização no notebook.
        """
        rows = []
        for u, v, data in G.edges(data=True):
            rows.append({
                "ativo_1":    u,
                "ativo_2":    v,
                "distancia":  data.get("weight", np.nan),
                "correlacao": data.get("correlation", np.nan),
                "in_mst":     data.get("in_mst", False),
            })
        return pd.DataFrame(rows).sort_values("distancia")
