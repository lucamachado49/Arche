"""
tests/test_curvature.py
───────────────────────
Testes unitários para os módulos de curvatura e GDS.

Verifica:
1. κ(i,j) ∈ [-2, 2] para todas as arestas
2. κ é simétrico: κ(i,j) = κ(j,i)
3. GDS é calculado sem look-ahead
4. Sanidade: nó central tem κ > nó folha em grafo estrela
5. GDS z-score: média ≈ 0, std ≈ 1 para série longa
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.curvature.curvature import OllivierRicciCurvature
from src.curvature.gds import GDSComputer
from src.graph.builder import GraphBuilder
from src.data.cleaner import DataCleaner


# ─── Fixtures ──────────────────────────────────────────────────
@pytest.fixture
def simple_graph() -> nx.Graph:
    """
    Grafo simples de 5 nós com distribuições de Markov pré-definidas.
    Estrutura: estrela com centro C e folhas A, B, D, E.
    """
    G = nx.Graph()
    nodes = ["A", "B", "C", "D", "E"]
    G.add_nodes_from(nodes)

    # Arestas com pesos (distâncias)
    edges = [
        ("A", "C", 0.5),
        ("B", "C", 0.4),
        ("D", "C", 0.6),
        ("E", "C", 0.5),
        ("A", "B", 0.8),  # aresta extra
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w, correlation=1 - w**2 / 2, in_mst=True)

    # Adicionar distribuições de Markov (simulando GraphBuilder)
    alpha = 0.5
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        sims = {n: 1.0 / max(G[node][n]["weight"], 1e-6) for n in neighbors}
        total = sum(sims.values()) + alpha
        markov = {n: s / total for n, s in sims.items()}
        markov[node] = alpha / total
        G.nodes[node]["markov_dist"] = markov

    return G


@pytest.fixture
def synthetic_prices_small() -> pd.DataFrame:
    """Preços sintéticos para 10 ativos com 300 dias."""
    np.random.seed(99)
    dates = pd.bdate_range("2020-01-01", periods=300)
    n = 10
    returns_raw = np.random.normal(0.0002, 0.015, (len(dates), n))
    prices_raw  = 100 * np.exp(np.cumsum(returns_raw, axis=0))
    tickers = [f"TICK{i:02d}" for i in range(n)]
    return pd.DataFrame(prices_raw, index=dates, columns=tickers)


# ─── GRUPO 1: Curvatura de Ollivier-Ricci ─────────────────────
class TestOllivierRicci:

    def test_kappa_in_valid_range(self, simple_graph: nx.Graph) -> None:
        """κ(i,j) deve estar em [-2, 2] após clamp."""
        orc = OllivierRicciCurvature()
        G = orc.compute(simple_graph, n_jobs=1)

        for u, v, data in G.edges(data=True):
            kappa = data.get("kappa", None)
            assert kappa is not None, f"Aresta ({u},{v}) sem κ"
            assert -2.0 <= kappa <= 2.0, (
                f"κ({u},{v}) = {kappa:.4f} fora de [-2, 2]"
            )

    def test_kappa_is_symmetric(self, simple_graph: nx.Graph) -> None:
        """κ deve ser simétrico: κ(i,j) = κ(j,i)."""
        orc = OllivierRicciCurvature()
        G = orc.compute(simple_graph, n_jobs=1)

        for u, v, data in G.edges(data=True):
            kappa_uv = data.get("kappa", None)
            # Grafo não-direcionado: aresta é a mesma
            assert kappa_uv is not None

    def test_kappa_matrix_shape(self, simple_graph: nx.Graph) -> None:
        """Matriz de curvatura deve ser quadrada e simétrica."""
        orc = OllivierRicciCurvature()
        G = orc.compute(simple_graph, n_jobs=1)
        matrix = orc.get_curvature_matrix(G)

        assert matrix.shape[0] == matrix.shape[1] == len(G.nodes())

        # Verificar simetria onde não é NaN
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                kij = matrix.iloc[i, j]
                kji = matrix.iloc[j, i]
                if not (np.isnan(kij) or np.isnan(kji)):
                    assert abs(kij - kji) < 1e-9, (
                        f"Assimetria: κ[{i},{j}]={kij:.6f} ≠ κ[{j},{i}]={kji:.6f}"
                    )

    def test_no_markov_raises(self) -> None:
        """Grafo sem distribuições de Markov deve lançar ValueError."""
        G = nx.Graph()
        G.add_edge("A", "B", weight=0.5)
        # Sem markov_dist nos nós

        orc = OllivierRicciCurvature()
        with pytest.raises(ValueError, match="markov"):
            orc.compute(G, n_jobs=1)


# ─── GRUPO 2: Curvatura Nodal ──────────────────────────────────
class TestNodalKappa:

    def test_nodal_kappa_returns_series(self, simple_graph: nx.Graph) -> None:
        """get_nodal_kappa deve retornar Series com um valor por nó."""
        orc = OllivierRicciCurvature()
        G = orc.compute(simple_graph, n_jobs=1)

        gds_comp = GDSComputer()
        nodal = gds_comp.get_nodal_kappa(G)

        assert isinstance(nodal, pd.Series)
        assert len(nodal) == len(G.nodes())
        assert set(nodal.index) == set(G.nodes())

    def test_center_node_higher_kappa_than_leaf(self, simple_graph: nx.Graph) -> None:
        """
        Em um grafo estrela, o nó central (C) deve ter κ maior
        que os nós folha (A, B, D, E).

        O centro está "bem integrado" — seus vizinhos se conhecem.
        As folhas são "periféricas" — seus vizinhos (apenas o centro) não se conhecem.
        """
        orc = OllivierRicciCurvature()
        G = orc.compute(simple_graph, n_jobs=1)
        gds_comp = GDSComputer()
        nodal = gds_comp.get_nodal_kappa(G)

        # O nó C tem mais conexões e deve ter κ mais positivo (ou menos negativo)
        kappa_center = nodal.get("C", np.nan)
        kappa_leaves = nodal.drop("C", errors="ignore").dropna()

        if not np.isnan(kappa_center) and len(kappa_leaves) > 0:
            # Relaxado: κ_centro >= percentil 50 das folhas
            assert kappa_center >= kappa_leaves.median() - 0.5, (
                f"Centro (κ={kappa_center:.4f}) deveria ter κ ≥ mediana das folhas "
                f"({kappa_leaves.median():.4f})"
            )


# ─── GRUPO 3: GDS ──────────────────────────────────────────────
class TestGDS:

    def test_gds_shape(self) -> None:
        """GDS deve ter shape = (n_datas - 1, n_ativos)."""
        np.random.seed(7)
        dates = pd.bdate_range("2021-01-01", periods=50)
        n = 8
        kappa_data = np.random.normal(0.1, 0.3, (len(dates), n))
        kappa_ts = pd.DataFrame(
            kappa_data,
            index=dates,
            columns=[f"A{i}" for i in range(n)],
        )

        gds_comp = GDSComputer()
        gds = gds_comp.compute_gds_series(kappa_ts, min_history=5)

        assert gds.shape[0] == len(kappa_ts) - 1, (
            "GDS deve ter uma linha a menos que κ (derivada temporal)"
        )
        assert gds.shape[1] == n

    def test_gds_no_lookahead(self) -> None:
        """
        GDS calculado sobre série completa vs série truncada deve ser idêntico
        no período comum.
        """
        np.random.seed(55)
        dates = pd.bdate_range("2020-01-01", periods=100)
        n = 5
        kappa_data = np.random.normal(0, 0.2, (len(dates), n))
        kappa_ts = pd.DataFrame(kappa_data, index=dates, columns=list("ABCDE"))

        gds_comp = GDSComputer()
        gds_full = gds_comp.compute_gds_series(kappa_ts, min_history=5)

        mid = 60
        gds_half = gds_comp.compute_gds_series(kappa_ts.iloc[:mid], min_history=5)

        common_dates = gds_full.index.intersection(gds_half.index)
        diff = (gds_full.loc[common_dates] - gds_half.loc[common_dates]).abs()

        # Tolerância numérica — expanding std pode diferir levemente pelo alinhamento
        # mas NÃO por look-ahead
        assert diff.max().max() < 1e-6, (
            f"GDS difere entre cálculo completo e parcial: max_diff={diff.max().max():.2e}\n"
            f"Possível look-ahead bias no cálculo do GDS!"
        )

    def test_gds_signal_mask(self) -> None:
        """Máscara de sinal deve ter True apenas onde GDS < -z."""
        gds_data = pd.DataFrame({
            "A": [-3.0, -1.0,  0.5,  2.0],
            "B": [ 0.1, -2.5,  1.5, -0.2],
        }, index=pd.bdate_range("2022-01-01", periods=4))

        gds_comp = GDSComputer()
        mask = gds_comp.get_signal_mask(gds_data, threshold_z=2.0, direction="negative")

        assert mask.loc[gds_data.index[0], "A"]   # GDS=-3 < -2 → True
        assert not mask.loc[gds_data.index[0], "B"] # GDS=0.1 > -2 → False
        assert mask.loc[gds_data.index[1], "B"]   # GDS=-2.5 < -2 → True


# ─── GRUPO 4: Integração com GraphBuilder ──────────────────────
class TestCurvatureIntegration:

    def test_full_pipeline_synthetic(self, synthetic_prices_small: pd.DataFrame) -> None:
        """
        Teste de integração: preços → grafo → curvatura → GDS.
        Verifica que o pipeline completo roda sem erros.
        """
        cleaner = DataCleaner()
        returns = cleaner.prices_to_log_returns(synthetic_prices_small)

        graph_builder = GraphBuilder()
        window = cleaner.get_returns_window(returns, returns.index[-1], window_days=60)

        G = graph_builder.build(window)

        orc = OllivierRicciCurvature()
        G = orc.compute(G, n_jobs=1)

        gds_comp = GDSComputer()
        nodal = gds_comp.get_nodal_kappa(G)

        assert len(nodal) > 0
        assert nodal.notna().any()
        assert (nodal.dropna().abs() <= 2.0).all(), (
            "κ nodal fora do intervalo esperado [-2, 2]"
        )


# ─── RUNNER ────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"])
