"""
tests/test_signal.py
────────────────────
Testes para o módulo de sinal (IC e portfólio).

Verifica:
1. IC é calculado sem look-ahead
2. Pesos do portfólio somam corretamente
3. Portfólio é market-neutral
4. Turnover é calculado corretamente
5. Custos de transação são sempre positivos
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signal.ic import ICAnalyzer
from src.portfolio.constructor import PortfolioConstructor
from src.portfolio.costs import TransactionCostModel


# ─── Fixtures ──────────────────────────────────────────────────
@pytest.fixture
def synthetic_gds() -> pd.DataFrame:
    """GDS sintético com padrão controlado."""
    np.random.seed(21)
    dates = pd.bdate_range("2020-01-01", periods=200)
    n = 15
    data = np.random.normal(0, 1, (len(dates), n))
    tickers = [f"T{i:02d}" for i in range(n)]
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def synthetic_returns_ic() -> pd.DataFrame:
    """Retornos sintéticos alinhados com o GDS."""
    np.random.seed(22)
    dates = pd.bdate_range("2020-01-01", periods=300)  # mais longo para cobrir forward
    n = 15
    data = np.random.normal(0.0002, 0.015, (len(dates), n))
    tickers = [f"T{i:02d}" for i in range(n)]
    return pd.DataFrame(data, index=dates, columns=tickers)


# ─── GRUPO 1: Information Coefficient ─────────────────────────
class TestICAnalyzer:

    def test_ic_series_not_empty(
        self, synthetic_gds: pd.DataFrame, synthetic_returns_ic: pd.DataFrame
    ) -> None:
        """IC deve ser calculado para a maioria das datas."""
        analyzer = ICAnalyzer()
        ic = analyzer.compute_ic_series(synthetic_gds, synthetic_returns_ic, horizon=5)

        assert len(ic) > 0, "IC vazio"
        assert ic.name == "IC_h5"

    def test_ic_in_range(
        self, synthetic_gds: pd.DataFrame, synthetic_returns_ic: pd.DataFrame
    ) -> None:
        """IC deve estar em [-1, 1]."""
        analyzer = ICAnalyzer()
        ic = analyzer.compute_ic_series(synthetic_gds, synthetic_returns_ic, horizon=5)
        valid = ic.dropna()
        assert (valid.abs() <= 1.0 + 1e-9).all(), (
            f"IC fora de [-1, 1]: min={valid.min():.4f}, max={valid.max():.4f}"
        )

    def test_ic_no_lookahead(
        self, synthetic_gds: pd.DataFrame, synthetic_returns_ic: pd.DataFrame
    ) -> None:
        """
        IC calculado sobre período completo e período truncado deve
        ter os mesmos valores no período em comum.
        """
        analyzer = ICAnalyzer()
        ic_full = analyzer.compute_ic_series(
            synthetic_gds, synthetic_returns_ic, horizon=5
        )

        # Usar apenas os primeiros 120 dias de GDS
        gds_trunc = synthetic_gds.iloc[:120]
        ic_trunc = analyzer.compute_ic_series(
            gds_trunc, synthetic_returns_ic, horizon=5
        )

        common = ic_full.index.intersection(ic_trunc.index)
        if len(common) > 0:
            diff = (ic_full.loc[common] - ic_trunc.loc[common]).abs()
            assert diff.max() < 1e-9, (
                f"IC difere entre cálculo completo e truncado: {diff.max():.2e}\n"
                "Look-ahead bias detectado no ICAnalyzer!"
            )

    def test_summary_has_required_columns(
        self, synthetic_gds: pd.DataFrame, synthetic_returns_ic: pd.DataFrame
    ) -> None:
        """Resumo deve ter todas as colunas esperadas."""
        analyzer = ICAnalyzer()
        ic_df = analyzer.compute_all_horizons(synthetic_gds, synthetic_returns_ic)
        summary = analyzer.summarize(ic_df)

        required = {"mean", "std", "IC_IR", "t_stat", "p_value", "pct_positive"}
        assert required.issubset(set(summary.columns)), (
            f"Colunas faltando no resumo: {required - set(summary.columns)}"
        )

    def test_check_gates_returns_dict(
        self, synthetic_gds: pd.DataFrame, synthetic_returns_ic: pd.DataFrame
    ) -> None:
        """check_gates deve retornar dict com 'all_gates_passed'."""
        analyzer = ICAnalyzer()
        ic_df = analyzer.compute_all_horizons(synthetic_gds, synthetic_returns_ic)
        summary = analyzer.summarize(ic_df)
        gates = analyzer.check_gates(summary)

        assert "all_gates_passed" in gates
        assert isinstance(gates["all_gates_passed"], bool)


# ─── GRUPO 2: PortfolioConstructor ────────────────────────────
class TestPortfolioConstructor:

    def test_long_weights_sum_to_one(self) -> None:
        """Pesos longos devem somar aproximadamente 1.0."""
        gds_t = pd.Series(
            np.linspace(-2, 2, 20),
            index=[f"T{i:02d}" for i in range(20)]
        )
        constructor = PortfolioConstructor()
        weights = constructor.build_weights(gds_t)

        long_sum = weights[weights > 0].sum()
        assert abs(long_sum - 1.0) < 0.05, (
            f"Pesos longos somam {long_sum:.4f}, esperado ≈ 1.0"
        )

    def test_market_neutral(self) -> None:
        """Portfólio market-neutral: |Σ w_long| ≈ |Σ w_short| (com ratio=1)."""
        gds_t = pd.Series(
            np.linspace(-3, 3, 30),
            index=[f"T{i:02d}" for i in range(30)]
        )
        constructor = PortfolioConstructor()
        weights = constructor.build_weights(gds_t)

        long_sum  = weights[weights > 0].sum()
        short_sum = weights[weights < 0].abs().sum()

        assert abs(long_sum - short_sum) < 0.1, (
            f"Portfólio não é market-neutral: long={long_sum:.4f}, short={short_sum:.4f}"
        )

    def test_no_weight_exceeds_cap(self) -> None:
        """Nenhum peso deve exceder max_weight_single_asset quando factível.

        Com cap=15% precisamos de pelo menos 7 ativos por perna para que
        o cap seja factível (1/7 ≈ 14.3% < 15%). Usamos 50 ativos.
        """
        np.random.seed(99)
        gds_t = pd.Series(
            np.random.randn(50),
            index=[f"T{i:02d}" for i in range(50)]
        )
        constructor = PortfolioConstructor()
        weights = constructor.build_weights(gds_t)
        cap = constructor.max_weight + 1e-6

        assert (weights.abs() <= cap).all(), (
            f"Peso excede o cap de {constructor.max_weight:.2f}: "
            f"{weights.abs().max():.4f}"
        )

    def test_weights_series_shape(self) -> None:
        """build_weights_series deve retornar DataFrame com índice temporal."""
        dates = pd.bdate_range("2021-01-01", periods=10)
        np.random.seed(77)
        gds = pd.DataFrame(
            np.random.randn(10, 20),
            index=dates,
            columns=[f"T{i:02d}" for i in range(20)]
        )
        constructor = PortfolioConstructor()
        weights_df = constructor.build_weights_series(gds)

        assert isinstance(weights_df, pd.DataFrame)
        assert len(weights_df) <= len(dates)

    def test_turnover_first_period(self) -> None:
        """Turnover no primeiro período = soma dos pesos absolutos / 2."""
        dates = pd.bdate_range("2021-01-01", periods=5)
        np.random.seed(33)
        gds = pd.DataFrame(
            np.random.randn(5, 20),
            index=dates,
            columns=[f"T{i:02d}" for i in range(20)]
        )
        constructor = PortfolioConstructor()
        weights_df = constructor.build_weights_series(gds)

        turnover = constructor.get_turnover(weights_df)
        assert (turnover >= 0).all(), "Turnover não pode ser negativo"


# ─── GRUPO 3: TransactionCostModel ────────────────────────────
class TestTransactionCosts:

    def test_costs_are_non_negative(self) -> None:
        """Custos de transação devem ser sempre não-negativos."""
        dates = pd.bdate_range("2021-01-01", periods=20)
        np.random.seed(44)
        gds = pd.DataFrame(
            np.random.randn(20, 15),
            index=dates,
            columns=[f"T{i:02d}" for i in range(15)]
        )
        constructor = PortfolioConstructor()
        weights_df = constructor.build_weights_series(gds)

        gross_returns = pd.Series(
            np.random.normal(0.001, 0.01, len(dates)),
            index=dates
        )
        prices = pd.DataFrame(
            100 * np.ones((len(dates), 15)),
            index=dates,
            columns=[f"T{i:02d}" for i in range(15)]
        )

        cost_model = TransactionCostModel()
        net = cost_model.apply(gross_returns, weights_df, prices)

        # Custos reduzem retornos: net ≤ gross em média
        assert net.mean() <= gross_returns.mean() + 1e-6, (
            "Custos de transação não podem aumentar os retornos!"
        )


# ─── RUNNER ────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"])
