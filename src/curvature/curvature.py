"""
src/curvature/curvature.py
──────────────────────────
Implementação da curvatura de Ollivier-Ricci discreta para grafos financeiros.

TEORIA
──────
Dado um grafo G = (V, E) com distribuições de Markov μᵢ em cada nó i,
a curvatura de Ollivier-Ricci na aresta (i,j) é:

    κ(i,j) = 1 − W₁(μᵢ, μⱼ) / d(i,j)

onde:
  - W₁ é a distância de Wasserstein-1 (Earth Mover's Distance)
  - d(i,j) é a distância métrica entre i e j no grafo
  - μᵢ é a distribuição de Markov no nó i (calculada em builder.py)

INTERPRETAÇÃO FINANCEIRA
─────────────────────────
  κ(i,j) > 0 : mercado "positivamente curvado" — os vizinhos de i e j
               são próximos entre si. Ativo bem integrado no cluster.
  κ(i,j) < 0 : mercado "negativamente curvado" — i e j são "pontes"
               separando clusters distintos. Ativo em posição geométrica
               de stress — associado a risco sistêmico (Sandhu et al., 2016).
  κ(i,j) = 0 : espaço "plano" — geometria euclideana local.

REFERÊNCIAS
───────────
  Ollivier (2009). Ricci curvature of Markov chains on metric spaces.
  Sandhu et al. (2016). Graph curvature for differentiating cancer networks.
  Lin et al. (2011). Ricci curvature of graphs.
"""
from __future__ import annotations

import time
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

try:
    import ot  # Python Optimal Transport (POT)
    _HAS_POT = True
except ImportError:
    _HAS_POT = False
    logger.warning(
        "POT (Python Optimal Transport) não instalado. "
        "Usando implementação alternativa de W₁. "
        "Execute: pip install POT"
    )

from src.data.config import get


class OllivierRicciCurvature:
    """
    Calcula a curvatura de Ollivier-Ricci para todos os pares de arestas
    de um grafo de correlações financeiras.

    As distribuições de Markov μᵢ devem já ter sido calculadas pelo
    GraphBuilder._add_markov_distributions() e armazenadas em G.nodes[i]["markov_dist"].

    Uso típico
    ----------
    orc = OllivierRicciCurvature()
    G_with_curvature = orc.compute(G)
    # G agora tem G[u][v]["kappa"] para cada aresta
    """

    def __init__(self) -> None:
        cfg = get("curvature")
        self.solver          = cfg.get("wasserstein_solver", "emd")
        self.sinkhorn_reg    = cfg.get("sinkhorn_reg", 0.1)
        self.max_lp_seconds  = cfg.get("max_lp_time_seconds", 30)

    # ──────────────────────────────────────────────────────────────
    # INTERFACE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def compute(
        self,
        G: nx.Graph,
        n_jobs: int = -1,
    ) -> nx.Graph:
        """
        Calcula κ(i,j) para todas as arestas do grafo.

        LOOK-AHEAD SAFETY:
        Este método opera apenas sobre o grafo G já construído com
        dados truncados em t. Não acessa datas — é look-ahead safe. ✓

        Parameters
        ----------
        G       : grafo com markov_dist em cada nó (saída de GraphBuilder)
        n_jobs  : paralelismo (-1 = todos os cores). Use 1 para debug.

        Returns
        -------
        nx.Graph com atributo "kappa" em cada aresta.
        κ ∈ [-1, 1] em geral, mas pode sair desse intervalo para
        distribuições não normalizadas — clampado em [-2, 2] por segurança.
        """
        t0 = time.perf_counter()

        # Validar que as distribuições de Markov existem
        missing_markov = [n for n in G.nodes() if "markov_dist" not in G.nodes[n]]
        if missing_markov:
            raise ValueError(
                f"{len(missing_markov)} nós sem distribuição de Markov. "
                f"Execute GraphBuilder._add_markov_distributions() primeiro."
            )

        edges = list(G.edges())
        n_edges = len(edges)

        logger.info(
            f"Calculando Ollivier-Ricci: {n_edges} arestas, "
            f"solver={self.solver}, n_jobs={n_jobs}"
        )

        # Pré-calcular lookup de distâncias no grafo (usada como ground cost)
        dist_lookup = self._precompute_distances(G)

        # Calcular curvatura em paralelo
        if n_jobs != 1 and n_edges > 20:
            kappas = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._edge_curvature)(G, u, v, dist_lookup)
                for u, v in edges
            )
        else:
            kappas = [
                self._edge_curvature(G, u, v, dist_lookup)
                for u, v in edges
            ]

        # Atribuir ao grafo
        for (u, v), kappa in zip(edges, kappas):
            G[u][v]["kappa"] = float(np.clip(kappa, -2.0, 2.0))

        elapsed = time.perf_counter() - t0
        kappa_vals = np.array([G[u][v]["kappa"] for u, v in edges])

        logger.info(
            f"Curvatura calculada em {elapsed:.2f}s | "
            f"κ médio={kappa_vals.mean():.4f}, "
            f"κ min={kappa_vals.min():.4f}, "
            f"κ max={kappa_vals.max():.4f}"
        )
        return G

    # ──────────────────────────────────────────────────────────────
    # CÁLCULO POR ARESTA
    # ──────────────────────────────────────────────────────────────
    def _edge_curvature(
        self,
        G: nx.Graph,
        u: str,
        v: str,
        dist_lookup: dict,
    ) -> float:
        """
        κ(u,v) = 1 − W₁(μᵤ, μᵥ) / d(u,v)

        Retorna NaN se os dados forem insuficientes.
        """
        d_uv = G[u][v].get("weight", 1.0)
        if d_uv <= 1e-9:
            # Distância zero = correlação perfeita = κ indefinida
            return np.nan

        mu_u = G.nodes[u]["markov_dist"]
        mu_v = G.nodes[v]["markov_dist"]

        w1 = self._wasserstein1(mu_u, mu_v, dist_lookup)
        if np.isnan(w1):
            return np.nan

        return 1.0 - w1 / d_uv

    def _wasserstein1(
        self,
        mu_u: dict[str, float],
        mu_v: dict[str, float],
        dist_lookup: dict,
    ) -> float:
        """
        Calcula W₁ entre duas distribuições de Markov discretas.

        As distribuições são dicionários {nó: probabilidade}.
        O ground cost é a distância d(i,j) no grafo (já precomputada).
        """
        # Unir os suportes das duas distribuições
        support = list(set(mu_u.keys()) | set(mu_v.keys()))
        n = len(support)

        if n == 0:
            return 0.0

        # Vetores de probabilidade alinhados
        p = np.array([mu_u.get(s, 0.0) for s in support], dtype=np.float64)
        q = np.array([mu_v.get(s, 0.0) for s in support], dtype=np.float64)

        # Normalizar (por segurança numérica)
        p = p / p.sum() if p.sum() > 1e-12 else np.ones(n) / n
        q = q / q.sum() if q.sum() > 1e-12 else np.ones(n) / n

        # Matriz de custos (distâncias no grafo)
        M = np.zeros((n, n), dtype=np.float64)
        for i, si in enumerate(support):
            for j, sj in enumerate(support):
                if i != j:
                    M[i, j] = dist_lookup.get((si, sj), dist_lookup.get((sj, si), 2.0))

        # Resolver com POT ou fallback
        if _HAS_POT:
            return self._solve_with_pot(p, q, M)
        else:
            return self._solve_greedy(p, q, M)

    def _solve_with_pot(
        self,
        p: np.ndarray,
        q: np.ndarray,
        M: np.ndarray,
    ) -> float:
        """Resolve W₁ via EMD exato ou Sinkhorn aproximado (POT library)."""
        try:
            if self.solver == "sinkhorn":
                w1 = ot.sinkhorn2(p, q, M, reg=self.sinkhorn_reg)[0]
            else:
                # EMD exato via Linear Programming
                w1 = ot.emd2(p, q, M)
            return float(w1)
        except Exception as e:
            logger.debug(f"POT falhou ({e}), usando fallback greedy")
            return self._solve_greedy(p, q, M)

    def _solve_greedy(
        self,
        p: np.ndarray,
        q: np.ndarray,
        M: np.ndarray,
    ) -> float:
        """
        Approximação de W₁ via algoritmo greedy de transporte ótimo.

        Não é exato para casos gerais, mas é uma boa aproximação quando
        as distribuições têm suportes pequenos (típico em grafos financeiros).

        Complexidade: O(n² log n) onde n = |suporte|.
        """
        p = p.copy()
        q = q.copy()
        n = len(p)

        # Ordenar pares por custo crescente (greedy)
        indices = np.argsort(M.ravel())
        total_cost = 0.0
        transported = 0.0
        target = min(p.sum(), q.sum())

        for idx in indices:
            if transported >= target - 1e-12:
                break
            i, j = divmod(idx, n)
            if M[i, j] == 0:
                continue
            flow = min(p[i], q[j])
            if flow > 1e-12:
                total_cost += flow * M[i, j]
                p[i] -= flow
                q[j] -= flow
                transported += flow

        return total_cost

    # ──────────────────────────────────────────────────────────────
    # UTILITÁRIOS
    # ──────────────────────────────────────────────────────────────
    def _precompute_distances(self, G: nx.Graph) -> dict:
        """
        Pré-computa todas as distâncias par-a-par no grafo.
        Necessário para o ground cost da EMD.

        Usa shortest path com pesos (distâncias de correlação).
        """
        try:
            # dict_of_dicts de shortest paths
            lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
            dist_lookup = {}
            for u, targets in lengths.items():
                for v, d in targets.items():
                    dist_lookup[(u, v)] = d
            return dist_lookup
        except Exception as e:
            logger.warning(f"Dijkstra falhou ({e}), usando distâncias de aresta diretas")
            dist_lookup = {}
            for u, v, data in G.edges(data=True):
                d = data.get("weight", 1.0)
                dist_lookup[(u, v)] = d
                dist_lookup[(v, u)] = d
            return dist_lookup

    def get_curvature_matrix(self, G: nx.Graph) -> pd.DataFrame:
        """
        Retorna a matriz de curvatura κ(i,j) como DataFrame.
        Útil para visualização e EDA.

        NaN onde não existe aresta direta.
        """
        nodes = sorted(G.nodes())
        n = len(nodes)
        node_idx = {n_: i for i, n_ in enumerate(nodes)}

        matrix = np.full((n, n), np.nan)
        np.fill_diagonal(matrix, 0.0)  # κ(i,i) = 0 por convenção

        for u, v, data in G.edges(data=True):
            kappa = data.get("kappa", np.nan)
            i, j = node_idx[u], node_idx[v]
            matrix[i, j] = kappa
            matrix[j, i] = kappa  # simétrico

        return pd.DataFrame(matrix, index=nodes, columns=nodes)
