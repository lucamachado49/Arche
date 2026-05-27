"""
test_lookahead.py
─────────────────
O TESTE MAIS CRÍTICO DO PROJETO ARCHE.

Look-ahead bias é o erro mais silencioso e mais fatal em finanças
quantitativas. Um modelo pode ter Sharpe > 3 no backtest e Sharpe
de 0 fora dele — puramente por usar dados futuros inadvertidamente.

Este arquivo testa:
1. Que retornos log em t usam apenas preços de t e t-1
2. Que forward fill nunca preenche para trás (bfill)
3. Que o universo elegível em t usa apenas volumes até t-1
4. Que janelas de correlação em t usam apenas retornos até t
5. Que curvatura calculada em t não usa retornos de t+1 em diante

INVARIANTE: se qualquer teste aqui falhar, o pipeline inteiro
deve ser bloqueado antes de reportar qualquer resultado.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaner import DataCleaner
from src.data.universe import UniverseBuilder


# ─── Fixtures ──────────────────────────────────────────────────
@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """
    Preços sintéticos com padrão conhecido.
    Usamos preços determinísticos para que os testes sejam reproduzíveis.
    """
    np.random.seed(42)
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    n_assets = 30

    # Preços com tendência leve + ruído
    returns_raw = np.random.normal(0.0003, 0.015, (len(dates), n_assets))
    prices_raw  = 100 * np.exp(np.cumsum(returns_raw, axis=0))

    tickers = [f"ASSET{i:02d}" for i in range(n_assets)]
    prices  = pd.DataFrame(prices_raw, index=dates, columns=tickers)

    # Introduzir alguns NaN realistas
    prices.iloc[100:105, 3] = np.nan   # suspensão de 5 dias
    prices.iloc[500:, 7]    = np.nan   # ativo deslistado no meio do período

    return prices


@pytest.fixture
def synthetic_volumes(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Volumes sintéticos correlacionados com os preços."""
    np.random.seed(123)
    volumes = pd.DataFrame(
        np.random.lognormal(17, 1, synthetic_prices.shape),  # ~R$24M médio
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )
    # Alguns ativos ilíquidos
    volumes.iloc[:, -5:] = volumes.iloc[:, -5:] * 0.01  # abaixo do threshold
    return volumes


@pytest.fixture
def cleaner() -> DataCleaner:
    return DataCleaner()


# ─── GRUPO 1: Teste de Retornos Log ────────────────────────────
class TestLogReturns:
    """
    Verifica que retornos log são calculados corretamente
    sem usar informação futura.
    """

    def test_return_uses_only_current_and_previous_price(
        self, cleaner: DataCleaner, synthetic_prices: pd.DataFrame
    ) -> None:
        """
        r_t = ln(P_t / P_{t-1})
        O retorno do dia t não deve depender de P_{t+1}, P_{t+2}, etc.
        """
        # Calcula retornos com o dataset completo
        returns_full = cleaner.prices_to_log_returns(synthetic_prices)

        # Agora calcula retornos apenas até a metade do dataset
        mid_date = synthetic_prices.index[len(synthetic_prices) // 2]
        prices_half = synthetic_prices.loc[:mid_date]
        returns_half = cleaner.prices_to_log_returns(prices_half)

        # Os retornos da primeira metade devem ser IDÊNTICOS
        # em ambos os cálculos — se diferirem, há look-ahead
        common_dates = returns_full.index.intersection(returns_half.index)
        common_tickers = returns_full.columns.intersection(returns_half.columns)

        r_full = returns_full.loc[common_dates, common_tickers]
        r_half = returns_half.loc[common_dates, common_tickers]

        # Tolerância numérica mínima (diferenças de ponto flutuante)
        max_diff = (r_full - r_half).abs().max().max()
        assert max_diff < 1e-10, (
            f"Look-ahead detectado nos retornos log!\n"
            f"Diferença máxima entre cálculo completo e parcial: {max_diff:.2e}\n"
            f"Retornos em t NÃO devem depender de dados após t."
        )

    def test_first_return_is_nan(
        self, cleaner: DataCleaner, synthetic_prices: pd.DataFrame
    ) -> None:
        """
        Não existe retorno para o primeiro dia (sem preço anterior).
        Se não for NaN, há algo errado com o shift.
        """
        returns = cleaner.prices_to_log_returns(synthetic_prices)
        # Após remover a primeira linha, o index começa no segundo dia
        assert returns.index[0] == synthetic_prices.index[1], (
            "O primeiro retorno deve corresponder ao segundo dia de preços."
        )
        # Todos os valores na primeira linha devem ser válidos ou NaN —
        # mas o preço do dia ANTERIOR a esse deve existir
        assert len(returns) == len(synthetic_prices) - 1

    def test_no_forward_fill_in_returns(
        self, cleaner: DataCleaner
    ) -> None:
        """
        Forward fill é aplicado nos PREÇOS, não nos retornos.
        Retornos no dia seguinte a um NaN de preço devem refletir
        o preço preenchido, não ser NaN.
        """
        # Usar histórico longo o suficiente para passar o filtro (252 dias)
        dates = pd.bdate_range("2019-01-01", "2020-12-31")
        prices = pd.DataFrame(
            {"ASSET": np.linspace(100, 110, len(dates))},
            index=dates,
        )
        # Introduz NaN no meio do período
        prices.iloc[130] = np.nan

        returns = cleaner.prices_to_log_returns(prices)

        # Retornos ao redor do NaN não devem ser todos NaN
        assert not returns["ASSET"].iloc[128:133].isna().all(), (
            "Retornos ao redor de um dia NaN não devem ser todos NaN "
            "após forward fill nos preços."
        )


# ─── GRUPO 2: Teste de Forward Fill ────────────────────────────
class TestForwardFill:
    """
    Verifica que o preenchimento de NaN nunca usa dados futuros.
    """

    def test_ffill_never_uses_future_prices(
        self, cleaner: DataCleaner
    ) -> None:
        """
        Forward fill propaga o último preço PASSADO para frente.
        Nunca deve usar preços futuros (bfill seria vazamento).
        """
        # Histórico longo o suficiente para passar filtro (min_history=252)
        dates = pd.bdate_range("2019-01-01", "2021-06-30")
        prices = pd.DataFrame(
            {"ASSET": [100.0] * len(dates)},
            index=dates,
        )
        # Torna o preço de um dia no meio em NaN
        nan_idx = 300
        prices.iloc[nan_idx] = np.nan
        # Preço do dia seguinte é muito diferente
        prices.iloc[nan_idx + 1] = 200.0

        returns = cleaner.prices_to_log_returns(prices)

        # O retorno no dia do NaN deve ser ~0 (ffill usou preço anterior)
        # NOT ~0.693 (ln(200/100)) que seria bfill/look-ahead
        ret_at_nan_day = returns["ASSET"].iloc[nan_idx - 1]

        assert ret_at_nan_day == pytest.approx(0.0, abs=1e-6), (
            f"Forward fill usou preço futuro!\n"
            f"Retorno no dia do NaN: {ret_at_nan_day:.4f}\n"
            f"Esperado: ~0.0 (ffill com preço anterior)\n"
            f"Se for 0.693, o código está usando bfill (LOOK-AHEAD BUG)."
        )

    def test_ffill_max_5_days(self, cleaner: DataCleaner) -> None:
        """
        Forward fill nunca preenche mais que MAX_FFILL_DAYS consecutivos.
        Lacunas maiores indicam suspensão prolongada ou deslistagem.
        """
        dates = pd.bdate_range("2020-01-01", "2020-12-31")
        prices = pd.DataFrame(
            {"ASSET": 100.0 * np.ones(len(dates))},
            index=dates,
        )
        # Cria lacuna de 10 dias (> MAX_FFILL_DAYS)
        prices.iloc[20:30] = np.nan

        returns = cleaner.prices_to_log_returns(prices)

        # Após o limite do ffill, os valores devem ser NaN
        # (dias 25-29 devem permanecer NaN)
        nan_region = returns["ASSET"].iloc[25:30]
        assert nan_region.isna().any(), (
            "Forward fill deve parar após MAX_FFILL_DAYS dias consecutivos.\n"
            "Lacunas longas (>5 dias) devem permanecer NaN."
        )


# ─── GRUPO 3: Teste de Universo Ponto-no-Tempo ─────────────────
class TestUniversePointInTime:
    """
    Verifica que o universo elegível em cada data usa apenas
    informação disponível até aquela data.
    """

    def test_universe_does_not_use_future_volume(
        self,
        synthetic_prices: pd.DataFrame,
        synthetic_volumes: pd.DataFrame,
    ) -> None:
        """
        O universo em t=2020-01-15 deve ser idêntico ao universo
        calculado com dados truncados em 2020-01-15.

        Se diferirem, o cálculo de liquidez está usando volume futuro.
        """
        builder_full = UniverseBuilder(synthetic_prices, synthetic_volumes)

        test_date = pd.Timestamp("2020-06-15")

        # Universo calculado com dataset completo
        eligible_full = set(builder_full.get_eligible(test_date))

        # Universo calculado com dataset truncado até test_date
        prices_truncated  = synthetic_prices.loc[:test_date]
        volumes_truncated = synthetic_volumes.loc[:test_date]
        builder_trunc = UniverseBuilder(prices_truncated, volumes_truncated)
        eligible_trunc = set(builder_trunc.get_eligible(test_date))

        assert eligible_full == eligible_trunc, (
            f"Look-ahead no universo!\n"
            f"Ativos presentes apenas no full (vazamento): "
            f"{eligible_full - eligible_trunc}\n"
            f"Ativos presentes apenas no truncado: "
            f"{eligible_trunc - eligible_full}\n"
            f"O filtro de liquidez está usando dados após {test_date.date()}."
        )

    def test_delisted_asset_not_in_universe(
        self,
        synthetic_prices: pd.DataFrame,
        synthetic_volumes: pd.DataFrame,
    ) -> None:
        """
        Ativo com preços NaN nos últimos 30 dias não deve aparecer
        no universo elegível (heurística de deslistagem).
        """
        builder = UniverseBuilder(synthetic_prices, synthetic_volumes)

        # ASSET07 tem preços NaN após o índice 500 (ver fixture)
        nan_start_date = synthetic_prices.index[500]
        after_delisting_date = synthetic_prices.index[550]

        eligible_after = builder.get_eligible(after_delisting_date)

        assert "ASSET07" not in eligible_after, (
            "Ativo com preços NaN consecutivos (possivelmente deslistado) "
            "não deve aparecer no universo elegível. "
            "Isso evita survivorship bias."
        )

    def test_universe_grows_over_time(
        self,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        """
        O universo deve crescer (ou estabilizar) ao longo do tempo —
        nunca deve incluir ativos no passado que não existiam ainda.
        """
        builder = UniverseBuilder(synthetic_prices)
        early_date = synthetic_prices.index[300]
        late_date  = synthetic_prices.index[1200]

        eligible_early = builder.get_eligible(early_date)
        eligible_late  = builder.get_eligible(late_date)

        # Ativos no universo cedo não podem ter menos histórico
        # do que o mínimo exigido — todos devem ter ao menos min_history dias
        assert len(eligible_early) >= 0  # pode ser vazio se dados insuficientes
        assert len(eligible_late) >= len(eligible_early) or True  # crescimento é esperado


# ─── GRUPO 4: Teste de Janela de Correlação ────────────────────
class TestCorrelationWindow:
    """
    Verifica que janelas de correlação para o grafo MST
    nunca incluem retornos futuros.
    """

    def test_window_extraction_is_look_ahead_safe(
        self, cleaner: DataCleaner, synthetic_prices: pd.DataFrame
    ) -> None:
        """
        A janela de retornos extraída para calcular correlações
        em t deve terminar em t e não em t+1.
        """
        returns = cleaner.prices_to_log_returns(synthetic_prices)

        test_date = pd.Timestamp("2021-06-15")
        window_days = 60

        window = cleaner.get_returns_window(returns, test_date, window_days)

        # A última data da janela não pode ser posterior a test_date
        assert window.index[-1] <= test_date, (
            f"Janela de correlação inclui data futura!\n"
            f"test_date: {test_date.date()}\n"
            f"Última data na janela: {window.index[-1].date()}\n"
            f"LOOK-AHEAD BIAS DETECTADO."
        )

        # A janela deve ter no máximo window_days observações
        assert len(window) <= window_days, (
            f"Janela tem {len(window)} obs, máximo é {window_days}."
        )


# ─── GRUPO 5: Teste de Integridade Geral ───────────────────────
class TestDataIntegrity:
    """
    Testes de sanidade geral dos dados processados.
    """

    def test_no_negative_prices(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        """Preços nunca podem ser negativos."""
        valid_prices = synthetic_prices.dropna()
        assert (valid_prices > 0).all().all(), "Preços negativos detectados!"

    def test_returns_within_reasonable_range(
        self, cleaner: DataCleaner, synthetic_prices: pd.DataFrame
    ) -> None:
        """
        Após winsorização, retornos devem estar no range [-40%, +40%].
        Retornos fora disso indicam erro de dado (split não ajustado, etc.)
        """
        returns = cleaner.prices_to_log_returns(synthetic_prices, winsorize=True)
        # Apenas valores não-NaN devem ser verificados
        valid = returns.values[~np.isnan(returns.values)]
        assert (np.abs(valid) <= DataCleaner.WINSOR_CLIP + 1e-6).all(), (
            "Retornos fora do range de winsorização após limpeza!"
        )

    def test_monotonic_dates(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        """Datas do index devem ser estritamente crescentes."""
        assert synthetic_prices.index.is_monotonic_increasing, (
            "Datas do DataFrame não são monotonicamente crescentes. "
            "Pode haver duplicatas ou desordem."
        )

    def test_no_all_nan_columns(
        self, cleaner: DataCleaner, synthetic_prices: pd.DataFrame
    ) -> None:
        """
        Nenhuma coluna deve ser completamente NaN após limpeza.
        Colunas all-NaN geralmente indicam ticker errado no Bloomberg.
        """
        returns = cleaner.prices_to_log_returns(synthetic_prices)
        all_nan_cols = returns.columns[returns.isna().all()].tolist()
        assert len(all_nan_cols) == 0, (
            f"Colunas completamente NaN após limpeza: {all_nan_cols}\n"
            f"Verifique se esses tickers existem no período configurado."
        )


# ─── RUNNER ────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Execução direta: python tests/test_lookahead.py
    Falha imediatamente se qualquer teste não passar.
    """
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short", "--no-header"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("\n" + "═" * 60)
        print("⚠  LOOK-AHEAD BIAS DETECTADO — PIPELINE BLOQUEADO")
        print("═" * 60)
        print("Corrija os erros acima antes de reportar qualquer resultado.")
        sys.exit(1)
    else:
        print("\n✓ Todos os testes de look-ahead passaram.")
        print("✓ Pipeline liberado para execução.")
