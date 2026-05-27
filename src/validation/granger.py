"""
src/validation/granger.py
──────────────────────────
Teste de causalidade de Granger: GDS → retorno futuro.

HIPÓTESE DE GRANGER
────────────────────
GDS "Granger-causa" o retorno futuro se as lags de GDS têm poder
preditivo incremental sobre o retorno, além das próprias lags do retorno.

H₀: GDS não Granger-causa retorno (coeficientes das lags GDS = 0)
H₁: GDS Granger-causa retorno (pelo menos um coeficiente ≠ 0)

Critério: p-value < 0.05 em pelo menos 1 lag configuration.

ATENÇÃO: Granger-causalidade ≠ causalidade verdadeira.
É uma relação de precedência temporal e poder preditivo. No contexto
do projeto, serve para demonstrar que o GDS não é uma transformação
trivial dos retornos passados.

REFERÊNCIAS
───────────
  Granger (1969). Investigating causal relations by econometric models.
  Hamilton (1994). Time Series Analysis. Cap. 11.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

from src.data.config import get


class GrangerCausalityTest:
    """
    Teste de causalidade de Granger entre GDS e retornos futuros.

    Uso típico
    ----------
    granger = GrangerCausalityTest()
    results = granger.test_gds_causes_returns(gds_ts, returns)
    summary = granger.summarize(results)
    """

    def __init__(self) -> None:
        cfg = get("validation")
        self.max_lags   = cfg.get("granger_max_lags", 10)
        self.max_pvalue = cfg.get("granger_max_pvalue", 0.05)

    # ──────────────────────────────────────────────────────────────
    # TESTE POR ATIVO
    # ──────────────────────────────────────────────────────────────
    def test_gds_causes_returns(
        self,
        gds_ts: pd.DataFrame,
        returns: pd.DataFrame,
        horizons: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Testa se GDS Granger-causa retorno para cada ativo e horizonte.

        Parameters
        ----------
        gds_ts   : DataFrame de GDS(t) por ativo
        returns  : DataFrame de retornos diários por ativo
        horizons : lags a testar (em dias). Default: [1, 5, 10]

        Returns
        -------
        pd.DataFrame com p-values por ativo e lag.
        """
        horizons = horizons or [1, 5, 10]
        results = []

        tickers = [t for t in gds_ts.columns if t in returns.columns]

        for ticker in tickers:
            gds = gds_ts[ticker].dropna()
            ret = returns[ticker].dropna()

            # Alinhar séries
            common = gds.index.intersection(ret.index)
            if len(common) < 60:
                continue

            gds_aligned = gds.loc[common]
            ret_aligned = ret.loc[common]

            # Teste de estacionaridade (necessário para Granger)
            gds_stationary = self._is_stationary(gds_aligned)
            ret_stationary = self._is_stationary(ret_aligned)

            if not gds_stationary:
                gds_aligned = gds_aligned.diff().dropna()
            if not ret_stationary:
                ret_aligned = ret_aligned.diff().dropna()

            # Re-alinhar após diferenciação
            common2 = gds_aligned.index.intersection(ret_aligned.index)
            if len(common2) < 30:
                continue

            data = pd.DataFrame({
                "return": ret_aligned.loc[common2],
                "gds":    gds_aligned.loc[common2],
            }).dropna()

            # Teste de Granger para cada número de lags
            ticker_result = {"ticker": ticker}
            for max_lag in range(1, min(self.max_lags + 1, len(data) // 10)):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gc_result = grangercausalitytests(
                            data[["return", "gds"]],
                            maxlag=max_lag,
                            verbose=False,
                        )
                    # Pegar p-value do F-test para o último lag
                    pval = gc_result[max_lag][0]["ssr_ftest"][1]
                    ticker_result[f"pval_lag{max_lag}"] = round(pval, 4)
                except Exception:
                    ticker_result[f"pval_lag{max_lag}"] = np.nan

            results.append(ticker_result)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).set_index("ticker")
        return df

    def test_aggregate(
        self,
        gds_ts: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> dict:
        """
        Testa causalidade de Granger usando o GDS e retorno AGREGADOS
        (médias do portfólio), não por ativo.

        Mais estável estatisticamente — N maior, menos ruído.
        """
        # Usar retorno médio cross-seccional como proxy do portfólio
        common_tickers = gds_ts.columns.intersection(returns.columns)
        gds_mean = gds_ts[common_tickers].mean(axis=1).dropna()
        ret_mean = returns[common_tickers].mean(axis=1).dropna()

        common_dates = gds_mean.index.intersection(ret_mean.index)
        data = pd.DataFrame({
            "return": ret_mean.loc[common_dates],
            "gds":    gds_mean.loc[common_dates],
        }).dropna()

        pvalues = {}
        for lag in range(1, self.max_lags + 1):
            if len(data) < lag * 10:
                break
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gc = grangercausalitytests(
                        data[["return", "gds"]],
                        maxlag=lag,
                        verbose=False,
                    )
                pvalues[lag] = gc[lag][0]["ssr_ftest"][1]
            except Exception:
                pvalues[lag] = np.nan

        min_pval = min(pvalues.values()) if pvalues else 1.0
        passed = min_pval < self.max_pvalue

        result = {
            "pvalues_by_lag": pvalues,
            "min_pvalue":     min_pval,
            "best_lag":       min(pvalues, key=pvalues.get) if pvalues else None,
            "passed":         passed,
        }

        if passed:
            logger.info(
                f"✅ Granger: GDS causa retorno (p={min_pval:.4f} < {self.max_pvalue})"
            )
        else:
            logger.warning(
                f"⚠️  Granger: evidência fraca (p_min={min_pval:.4f} ≥ {self.max_pvalue})"
            )
        return result

    def summarize(self, results_df: pd.DataFrame) -> dict:
        """
        Resume os resultados: % de ativos com GDS Granger-causando retorno.
        """
        if results_df.empty:
            return {"pct_significant": 0.0, "n_tickers": 0}

        pval_cols = [c for c in results_df.columns if c.startswith("pval_")]
        if not pval_cols:
            return {"pct_significant": 0.0, "n_tickers": len(results_df)}

        # Ativo é "significativo" se pelo menos 1 lag tem p < threshold
        significant = (results_df[pval_cols] < self.max_pvalue).any(axis=1)
        pct_sig = significant.mean()

        logger.info(
            f"Granger (por ativo): {pct_sig:.1%} dos ativos com GDS "
            f"Granger-causando retorno (p < {self.max_pvalue})"
        )
        return {
            "pct_significant": round(pct_sig, 4),
            "n_tickers": len(results_df),
            "n_significant": int(significant.sum()),
        }

    # ──────────────────────────────────────────────────────────────
    # AUXILIARES
    # ──────────────────────────────────────────────────────────────
    def _is_stationary(self, series: pd.Series, alpha: float = 0.05) -> bool:
        """
        Teste ADF de estacionaridade.
        Retorna True se a série é estacionária (rejeita raiz unitária).
        """
        try:
            result = adfuller(series.dropna(), autolag="AIC")
            return result[1] < alpha  # p-value < alpha → rejeita H0 → estacionária
        except Exception:
            return True  # assume estacionária se teste falhar
