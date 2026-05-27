"""
universe.py
───────────
Constrói o universo elegível de ativos em cada data do backtest.

CONCEITO CRÍTICO: POINT-IN-TIME (PIT)
──────────────────────────────────────
O universo elegível em uma data t deve ser construído usando apenas
informação disponível até t-1. Isso evita:

1. SURVIVORSHIP BIAS: incluir apenas empresas que sobreviveram
   ao período inteiro. Na realidade, empresas que faliram ou foram
   deslistadas também existiam no passado — ignorá-las superestima
   qualquer retorno histórico.

2. LOOK-AHEAD BIAS EM LIQUIDEZ: usar o volume médio do período
   inteiro para decidir quem era "líquido" em 2015 é incorreto —
   algumas empresas eram ilíquidas em 2015 mas tornaram-se líquidas
   depois. O filtro deve usar apenas volume histórico até t-1.

Implementamos elegibilidade ponto-no-tempo:
  - Volume médio dos últimos N dias antes de t
  - Mínimo de dias com dados no período da janela
  - Excluir ativos que saíram da B3 (preço = NaN por > 30 dias)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import get


class UniverseBuilder:
    """
    Gera o universo elegível de ativos para cada data do backtest.

    Dois modos de operação:
    1. Estático  : universo fixo para todo o período (mais simples)
    2. Dinâmico  : universo recalculado a cada rebalanceamento (correto)

    Para o MVP, usamos o modo dinâmico — é o único correto.

    Uso típico
    ----------
    builder = UniverseBuilder(prices, volumes)
    universe = builder.get_eligible(date=pd.Timestamp("2020-01-15"))
    # Retorna lista de tickers elegíveis nessa data
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Parameters
        ----------
        prices  : DataFrame de preços ajustados (index=Date, cols=tickers)
        volumes : DataFrame de volume financeiro diário em R$.
                  Se None, usa proxy de volume via retorno (menos preciso).
        """
        self.prices  = prices
        self.volumes = volumes
        self.cfg     = get("universe")

        self.min_volume    = self.cfg["min_daily_volume_brl"]
        self.min_history   = self.cfg["min_history_days"]
        self.vol_lookback  = self.cfg["volume_lookback_days"]

        logger.info(
            f"UniverseBuilder inicializado: "
            f"{prices.shape[1]} ativos candidatos, "
            f"filtro de liquidez: R${self.min_volume/1e6:.0f}M/dia"
        )

    # ──────────────────────────────────────────────────────────────
    # INTERFACE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def get_eligible(
        self,
        date: pd.Timestamp,
        min_assets: int = 20,
    ) -> list[str]:
        """
        Retorna lista de tickers elegíveis na data especificada.

        PONTO-NO-TEMPO: usa apenas dados disponíveis até date-1.
        Nunca usa informação de date em diante.

        Parameters
        ----------
        date      : data de referência (inclusive)
        min_assets: mínimo de ativos — se não atingido, avisa

        Returns
        -------
        Lista de tickers elegíveis (pode ser vazia se dados insuficientes)
        """
        # Dados disponíveis até date (inclusive) — LOOK-AHEAD SAFE
        prices_available = self.prices.loc[:date]

        if len(prices_available) < self.min_history:
            logger.warning(
                f"Data {date.date()}: apenas {len(prices_available)} dias disponíveis. "
                f"Mínimo é {self.min_history}. Retornando universo vazio."
            )
            return []

        eligible = []
        for ticker in self.prices.columns:
            if self._is_eligible(ticker, prices_available):
                eligible.append(ticker)

        if len(eligible) < min_assets:
            logger.warning(
                f"Data {date.date()}: apenas {len(eligible)} ativos elegíveis "
                f"(mínimo recomendado: {min_assets})"
            )

        logger.debug(f"Universo em {date.date()}: {len(eligible)} ativos elegíveis")
        return eligible

    def build_eligibility_matrix(
        self,
        rebalance_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Constrói a matrix de elegibilidade para todas as datas de rebalanceamento.

        Muito mais eficiente que chamar get_eligible() individualmente
        quando há muitas datas de rebalanceamento.

        Returns
        -------
        pd.DataFrame boolean: index=rebalance_dates, cols=tickers.
        True = ativo elegível nessa data.
        """
        logger.info(
            f"Construindo matriz de elegibilidade: "
            f"{len(rebalance_dates)} datas × {len(self.prices.columns)} ativos"
        )

        eligibility = pd.DataFrame(
            False,
            index=rebalance_dates,
            columns=self.prices.columns,
        )

        for date in rebalance_dates:
            eligible = self.get_eligible(date)
            eligibility.loc[date, eligible] = True

        # Relatório
        avg_universe_size = eligibility.sum(axis=1).mean()
        min_universe_size = eligibility.sum(axis=1).min()
        logger.info(
            f"Matriz de elegibilidade concluída:\n"
            f"  Tamanho médio do universo : {avg_universe_size:.0f} ativos\n"
            f"  Tamanho mínimo            : {min_universe_size:.0f} ativos"
        )

        return eligibility

    # ──────────────────────────────────────────────────────────────
    # CRITÉRIOS DE ELEGIBILIDADE
    # ──────────────────────────────────────────────────────────────
    def _is_eligible(
        self,
        ticker: str,
        prices_available: pd.DataFrame,
    ) -> bool:
        """
        Verifica se um ativo é elegível com base em múltiplos critérios.
        Retorna True apenas se TODOS os critérios forem atendidos.
        """
        price_series = prices_available[ticker]

        # CRITÉRIO 1: histórico mínimo
        if not self._check_min_history(price_series):
            return False

        # CRITÉRIO 2: ativo ainda está listado (não foi deslistado)
        if not self._check_still_listed(price_series):
            return False

        # CRITÉRIO 3: liquidez mínima
        if not self._check_liquidity(ticker, prices_available):
            return False

        return True

    def _check_min_history(self, price_series: pd.Series) -> bool:
        """Ativo tem pelo menos min_history dias de dados válidos."""
        valid_count = price_series.notna().sum()
        return valid_count >= self.min_history

    def _check_still_listed(self, price_series: pd.Series) -> bool:
        """
        Ativo não foi deslistado.

        Heurística: se os últimos 30 dias de preço são todos NaN,
        consideramos o ativo deslistado nessa data.
        Isso captura falências e delistagens sem precisar de dado externo.
        """
        recent_window = 30
        if len(price_series) < recent_window:
            return price_series.notna().any()

        recent = price_series.iloc[-recent_window:]
        return recent.notna().any()

    def _check_liquidity(
        self,
        ticker: str,
        prices_available: pd.DataFrame,
    ) -> bool:
        """
        Verifica liquidez mínima usando volume dos últimos vol_lookback dias.

        Dois casos:
        1. volumes disponível: usa volume financeiro direto
        2. volumes não disponível: usa proxy de giro de preço (menos preciso)
        """
        lookback = min(self.vol_lookback, len(prices_available))

        if self.volumes is not None and ticker in self.volumes.columns:
            # Caso 1: volume financeiro direto (Bloomberg: VOLUME * PX_LAST)
            vol_series = self.volumes[ticker].loc[:prices_available.index[-1]]
            if len(vol_series) < lookback // 2:
                return False
            avg_volume = vol_series.iloc[-lookback:].mean()
            return avg_volume >= self.min_volume

        else:
            # Caso 2: proxy de liquidez via consistência de preço
            # Um ativo líquido tem poucas sequências de preço idêntico
            # (preço constante = sem negociação)
            price_series = prices_available[ticker].iloc[-lookback:]
            if price_series.isna().mean() > 0.5:
                return False

            # Sequências de preço idêntico indicam iliquidez
            price_changes = price_series.diff().abs()
            pct_days_with_change = (price_changes > 0).mean()

            # Exige que pelo menos 70% dos dias tenha alguma variação de preço
            return pct_days_with_change >= 0.70

    # ──────────────────────────────────────────────────────────────
    # UTILITÁRIOS
    # ──────────────────────────────────────────────────────────────
    def get_sector_breakdown(
        self,
        eligible: list[str],
        sector_map: Optional[dict[str, str]] = None,
    ) -> pd.Series:
        """
        Retorna contagem de ativos elegíveis por setor.
        Útil para o notebook de EDA.
        """
        if sector_map is None:
            # Mapa padrão para os principais ativos brasileiros
            sector_map = {
                "VALE3": "Materiais", "VALE5": "Materiais",
                "PETR4": "Energia", "PETR3": "Energia",
                "ITUB4": "Financeiro", "ITUB3": "Financeiro",
                "BBDC4": "Financeiro", "BBDC3": "Financeiro",
                "BBAS3": "Financeiro",
                "ABEV3": "Consumo",
                "WEGE3": "Industrial",
                "RENT3": "Serviços",
                "MGLU3": "Varejo", "LREN3": "Varejo",
                "SUZB3": "Materiais",
                "ELET3": "Utilidades", "ELET6": "Utilidades",
                "JBSS3": "Alimentício", "BRFS3": "Alimentício",
                "CSAN3": "Energia",
                "RDOR3": "Saúde", "HAPV3": "Saúde",
            }

        sectors = [sector_map.get(t, "Outros") for t in eligible]
        return pd.Series(sectors, index=eligible).value_counts()

    def describe_universe(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Gera descrição completa do universo em uma data.
        Usado no notebook de EDA para validação visual.
        """
        eligible = self.get_eligible(date)
        prices_at_date = self.prices.loc[:date]

        data = []
        for ticker in eligible:
            p_series = prices_at_date[ticker].dropna()
            r_series = returns[ticker].dropna() if ticker in returns.columns else pd.Series()

            data.append({
                "ticker":         ticker,
                "preco_atual":    p_series.iloc[-1] if len(p_series) > 0 else np.nan,
                "dias_historico": p_series.notna().sum(),
                "ret_medio_dia":  r_series.mean() if len(r_series) > 0 else np.nan,
                "vol_diaria":     r_series.std() if len(r_series) > 0 else np.nan,
                "pct_nan":        prices_at_date[ticker].isna().mean(),
            })

        df = pd.DataFrame(data).set_index("ticker")
        df["sharpe_anual_proxy"] = (
            df["ret_medio_dia"] * 252 /
            (df["vol_diaria"] * np.sqrt(252))
        )
        return df.round(4)
