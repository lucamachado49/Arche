"""
src/backtest/walkforward.py
────────────────────────────
Engine de backtest walk-forward para o Projeto Arche.

METODOLOGIA WALK-FORWARD
─────────────────────────
O backtest walk-forward evita overfitting ao nunca usar o mesmo
período para treinar e testar o modelo.

Estrutura de janelas:
  |──────── TREINO (3 anos) ────────|── TESTE (6 meses) ──|
                    |──────── TREINO (3 anos) ────────|── TESTE (6 meses) ──|
                                      ...

Parâmetros (settings.yaml):
  train_years: 3       # janela de treino
  test_months: 6       # janela de teste
  step_months: 3       # avanço entre iterações (overlap de 50%)

LOOK-AHEAD GARANTIDO
──────────────────────
Cada fold de teste usa exclusivamente dados após o período de treino.
O GDS é gerado apenas com dados do período de treino + histórico até t.
Os pesos de t usam GDS(t) que foi calculado com dados até t. ✓
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data.config import get
from src.portfolio.constructor import PortfolioConstructor
from src.portfolio.costs import TransactionCostModel
from .metrics import PerformanceMetrics


@dataclass
class WalkForwardFold:
    """Representa um fold do walk-forward."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    gross_returns: pd.Series
    net_returns: pd.Series
    weights: pd.DataFrame
    metrics: pd.Series


class WalkForwardBacktest:
    """
    Engine de backtest walk-forward para o portfólio GDS long-short.

    Uso típico
    ----------
    engine = WalkForwardBacktest()
    results = engine.run(
        returns=returns,
        kappa_ts=kappa_ts,
        gds_ts=gds_ts,
        prices=prices,
    )
    full_net_returns = engine.get_combined_returns(results)
    pm = PerformanceMetrics(full_net_returns)
    print(pm.summary())
    """

    def __init__(self) -> None:
        cfg_bt = get("backtest")
        self.train_years = cfg_bt.get("train_years", 3)
        self.test_months = cfg_bt.get("test_months", 6)
        self.step_months = cfg_bt.get("step_months", 3)
        self.oos_start   = pd.Timestamp(cfg_bt.get("out_of_sample_start", "2019-01-01"))

        # Frequência de rebalanceamento. O GDS chega em alta frequência
        # (semanal), mas só negociamos na frequência do horizonte do alpha
        # (mensal). Ver _subsample_rebalance.
        self.rebalance_frequency = get("portfolio", "rebalance_frequency", "monthly")

        # Suavização EWMA causal do GDS (em períodos da grade semanal). O alpha
        # de curvatura é lento; o Δκ de 1 semana é ruído. Ver settings.signal.
        self.gds_smoothing_span = get("signal", "gds_smoothing_span", 1)

        self.constructor  = PortfolioConstructor()
        self.cost_model   = TransactionCostModel()

    # ──────────────────────────────────────────────────────────────
    # INTERFACE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def run(
        self,
        returns: pd.DataFrame,
        gds_ts: pd.DataFrame,
        prices: pd.DataFrame,
        eligible_universe: Optional[dict[pd.Timestamp, list[str]]] = None,
        benchmark: Optional[pd.Series] = None,
        risk_free: Optional[pd.Series] = None,
    ) -> list[WalkForwardFold]:
        """
        Executa o backtest walk-forward completo.

        Parameters
        ----------
        returns          : retornos diários por ativo (look-ahead safe)
        gds_ts           : GDS por data e ativo (gerado pelo SignalGenerator)
        prices           : preços para cálculo de custos
        eligible_universe: universo elegível por data (opcional)
        benchmark        : retornos do benchmark (IBOV) para IR
        risk_free        : taxa livre de risco diária (Selic/252)

        Returns
        -------
        Lista de WalkForwardFold com resultados por janela de teste.
        """
        # Suavização EWMA causal do GDS (look-ahead safe: ewm só olha p/ trás).
        # Feita ANTES do fold p/ que cada teste herde o histórico suavizado.
        if self.gds_smoothing_span and self.gds_smoothing_span > 1:
            gds_ts = gds_ts.ewm(span=self.gds_smoothing_span, adjust=False).mean()
            logger.info(f"GDS suavizado: EWMA span={self.gds_smoothing_span}")

        folds = self._get_folds(returns.index)
        logger.info(f"Walk-forward: {len(folds)} folds × {self.test_months} meses de teste")

        results = []

        for fold_dates in folds:
            train_start, train_end, test_start, test_end = fold_dates
            fold_id = len(results) + 1

            logger.info(
                f"Fold {fold_id}: treino {train_start.date()}→{train_end.date()} | "
                f"teste {test_start.date()}→{test_end.date()}"
            )

            # 1. GDS apenas no período de teste (usando historico até cada data)
            gds_test = gds_ts.loc[
                (gds_ts.index >= test_start) & (gds_ts.index <= test_end)
            ]

            if gds_test.empty:
                logger.warning(f"Fold {fold_id}: GDS vazio no período de teste.")
                continue

            # 1b. Reamostrar para a frequência de rebalanceamento (mensal).
            # Negociar o sinal mensal no horizonte em que ele tem poder
            # preditivo (h≈21), em vez de a cada data semanal de GDS.
            gds_test = self._subsample_rebalance(gds_test)

            # 2. Construir pesos no período de teste
            eligible_test = None
            if eligible_universe is not None:
                eligible_test = {
                    d: v for d, v in eligible_universe.items()
                    if test_start <= d <= test_end
                }

            weights = self.constructor.build_weights_series(
                gds_test, eligible_test, returns=returns
            )

            if weights.empty:
                logger.warning(f"Fold {fold_id}: pesos vazios.")
                continue

            # 3. Calcular retornos brutos — RESTRITOS à janela de teste.
            # Sem o slice, o ffill dos pesos vazaria a posição do último
            # rebalanceamento do fold até o fim da amostra (look-ahead de
            # holding). A combinação posterior (keep="last") garante que
            # cada fold contribua apenas com seus ~step_months frescos.
            gross_returns = self._compute_gross_returns(returns, weights)
            gross_returns = gross_returns.loc[test_start:test_end]

            if gross_returns.empty:
                continue

            # 4. Aplicar custos de transação
            net_returns = self.cost_model.apply(gross_returns, weights, prices)

            # 5. Calcular métricas
            pm = PerformanceMetrics(
                returns=net_returns,
                benchmark=benchmark,
                risk_free=risk_free,
            )

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                gross_returns=gross_returns,
                net_returns=net_returns,
                weights=weights,
                metrics=pm.summary(),
            )
            results.append(fold)

            logger.info(
                f"  Sharpe líquido: {pm.sharpe_ratio():.4f} | "
                f"Retorno anual: {pm.annualized_return():.2%} | "
                f"MaxDD: {pm.max_drawdown():.2%}"
            )

        # Gate Sprint 5
        self._check_sprint5_gate(results)

        return results

    # ──────────────────────────────────────────────────────────────
    # CAMINHO CONTÍNUO (sem artefato de recompra por fold)
    # ──────────────────────────────────────────────────────────────
    def run_continuous(
        self,
        returns: pd.DataFrame,
        gds_ts: pd.DataFrame,
        prices: pd.DataFrame,
        eligible_universe: Optional[dict[pd.Timestamp, list[str]]] = None,
    ) -> dict[str, pd.Series]:
        """
        Backtest em UMA passada contínua sobre todo o OOS.

        Por que existe: o sinal GDS é causal e NÃO há modelo treinado por fold
        (o "treino" só garante histórico mínimo). Logo, recortar em folds
        sobrepostos e cobrar custo de transação no início de CADA fold
        equivale a recomprar o book inteiro a cada `step_months` — um custo
        que não existe ao vivo, onde a posição é carregada. Aqui o book é
        construído uma vez, com hysteresis contínua, e o custo é cobrado só
        sobre o turnover REAL entre rebalanceamentos consecutivos.

        Retorna {"gross": ..., "net": ..., "weights": ...} (este último o
        DataFrame de pesos por data de rebalanceamento, p/ diagnóstico/custo).
        """
        if self.gds_smoothing_span and self.gds_smoothing_span > 1:
            gds_ts = gds_ts.ewm(span=self.gds_smoothing_span, adjust=False).mean()

        gds_oos = gds_ts.loc[gds_ts.index >= self.oos_start]
        gds_oos = self._subsample_rebalance(gds_oos)
        if gds_oos.empty:
            raise RuntimeError("GDS vazio no período OOS após reamostragem.")

        weights = self.constructor.build_weights_series(
            gds_oos, eligible_universe, returns=returns
        )

        gross = self._compute_gross_returns(returns, weights)
        gross = gross.loc[gross.index >= weights.index[0]]
        net = self.cost_model.apply(gross, weights, prices)

        return {"gross": gross, "net": net, "weights": weights}

    def fold_summary_from_series(self, net: pd.Series) -> pd.DataFrame:
        """
        Métricas por janela (mesmas datas dos folds) recortadas da série
        contínua de retornos líquidos — sem o artefato de recompra por fold.
        Serve só para inspecionar a estabilidade do sinal no tempo.
        """
        rows = []
        for i, (_, _, test_start, test_end) in enumerate(self._get_folds(net.index), 1):
            seg = net.loc[test_start:test_end]
            if seg.empty:
                continue
            pm = PerformanceMetrics(seg)
            rows.append({
                "fold":          i,
                "teste_inicio":  test_start.date(),
                "teste_fim":     test_end.date(),
                "sharpe":        round(pm.sharpe_ratio(), 4),
                "retorno_anual": round(pm.annualized_return(), 4),
                "max_drawdown":  round(pm.max_drawdown(), 4),
                "win_rate":      round(pm.win_rate(), 4),
                "n_dias":        len(seg),
            })
        return pd.DataFrame(rows).set_index("fold")

    # ──────────────────────────────────────────────────────────────
    # CONSOLIDAÇÃO DE RESULTADOS
    # ──────────────────────────────────────────────────────────────
    def get_combined_returns(
        self,
        results: list[WalkForwardFold],
        net: bool = True,
    ) -> pd.Series:
        """
        Consolida os retornos de todos os folds em uma série contínua.

        Os folds de teste se SOBREPÕEM (step=3m < teste=6m). Para evitar
        contagem dupla, em datas cobertas por mais de um fold mantemos o
        retorno do fold MAIS RECENTE (keep="last") — o de treino mais novo
        e ainda look-ahead-safe. Na prática, cada fold contribui apenas
        com seus primeiros ~step_months de teste.
        """
        series = []
        for fold in results:
            r = fold.net_returns if net else fold.gross_returns
            series.append(r)

        if not series:
            return pd.Series(dtype=float, name="portfolio_returns")

        combined = pd.concat(series).sort_index()
        combined.name = "portfolio_returns"
        return combined[~combined.index.duplicated(keep="last")]

    def get_fold_summary(self, results: list[WalkForwardFold]) -> pd.DataFrame:
        """
        Tabela resumo de métricas por fold.
        Útil para detectar degradação do sinal ao longo do tempo.
        """
        rows = []
        for fold in results:
            pm = PerformanceMetrics(fold.net_returns)
            rows.append({
                "fold":         fold.fold_id,
                "teste_inicio": fold.test_start.date(),
                "teste_fim":    fold.test_end.date(),
                "sharpe":       round(pm.sharpe_ratio(), 4),
                "retorno_anual": round(pm.annualized_return(), 4),
                "max_drawdown": round(pm.max_drawdown(), 4),
                "win_rate":     round(pm.win_rate(), 4),
                "n_dias":       len(fold.net_returns),
            })
        return pd.DataFrame(rows).set_index("fold")

    # ──────────────────────────────────────────────────────────────
    # UTILITÁRIOS PRIVADOS
    # ──────────────────────────────────────────────────────────────
    def _subsample_rebalance(self, gds_test: pd.DataFrame) -> pd.DataFrame:
        """
        Reduz a grade de GDS à frequência de rebalanceamento.

        O GDS é estimado semanalmente, mas o alpha só vive em h≈21 (mensal).
        Rebalancear a cada GDS semanal quadruplica o turnover para perseguir
        ruído (h=1,5,10 são nulos). Aqui mantemos apenas a ÚLTIMA data de
        GDS de cada período — o sinal mais fresco disponível, look-ahead
        safe (usa só dados ≤ data escolhida).

        - "monthly": última data de GDS de cada mês-calendário.
        - "weekly"/"daily": sem reamostragem (GDS já é semanal).
        """
        freq = (self.rebalance_frequency or "monthly").lower()
        if freq not in ("monthly", "month", "m"):
            return gds_test
        if gds_test.empty:
            return gds_test

        # Última data observada por mês-calendário (sinal mais recente do mês)
        keep = (
            gds_test.index.to_series()
            .groupby([gds_test.index.year, gds_test.index.month])
            .max()
        )
        return gds_test.loc[gds_test.index.isin(keep.values)]

    def _get_folds(
        self,
        date_index: pd.DatetimeIndex,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Gera os pares (train_start, train_end, test_start, test_end).

        LOOK-AHEAD: test_start é sempre posterior a train_end. ✓
        """
        folds = []
        test_start = self.oos_start
        max_date = date_index[-1]

        while test_start < max_date:
            # Período de treino: train_years antes do teste
            train_end   = test_start - pd.Timedelta(days=1)
            train_start = test_start - pd.DateOffset(years=self.train_years)
            test_end    = test_start + pd.DateOffset(months=self.test_months)
            test_end    = min(test_end, max_date)

            # Verificar se há dados suficientes
            train_days = (date_index >= train_start) & (date_index <= train_end)
            if train_days.sum() < 252:
                logger.warning(f"Fold ignorado: dados insuficientes em {test_start.date()}")
                test_start += pd.DateOffset(months=self.step_months)
                continue

            folds.append((train_start, train_end, test_start, test_end))
            test_start += pd.DateOffset(months=self.step_months)

        return folds

    def _compute_gross_returns(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """
        Calcula retorno bruto do portfólio.

        r_portfolio(t) = Σ_i w_i(t_prev) * r_i(t)

        onde t_prev é a última data de rebalanceamento antes de t.
        Os pesos são mantidos constantes entre rebalanceamentos.

        LOOK-AHEAD: pesos de t_prev (passado) multiplicados por retornos de t. ✓
        """
        # Expandir pesos para frequência diária (hold até próximo rebalanceamento)
        all_dates = returns.index
        weights_daily = weights.reindex(all_dates).ffill()

        # Interseção de colunas
        common_tickers = returns.columns.intersection(weights_daily.columns)
        if len(common_tickers) == 0:
            return pd.Series(dtype=float)

        r = returns[common_tickers]
        w = weights_daily[common_tickers].shift(1)  # shift: pesos de ontem × retorno de hoje

        # Retorno do portfólio = produto interno
        portfolio_returns = (r * w).sum(axis=1)
        return portfolio_returns.dropna()

    def _check_sprint5_gate(self, results: list[WalkForwardFold]) -> None:
        """Gate de saída do Sprint 5: Sharpe > 0.5 líquido de custos."""
        if not results:
            logger.error("Sprint 5 Gate: FALHOU — nenhum fold de teste concluído.")
            return

        combined = self.get_combined_returns(results, net=True)
        pm = PerformanceMetrics(combined)
        sharpe = pm.sharpe_ratio()
        target = 0.5

        if sharpe > target:
            logger.info(
                f"✅ Sprint 5 Gate: PASSOU — Sharpe líquido={sharpe:.4f} > {target}"
            )
        else:
            logger.warning(
                f"❌ Sprint 5 Gate: FALHOU — Sharpe líquido={sharpe:.4f} ≤ {target}\n"
                f"   Revisar: markov_alpha, gds_threshold_sigma, janela de correlação."
            )
