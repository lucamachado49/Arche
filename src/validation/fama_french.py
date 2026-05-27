"""
src/validation/fama_french.py
──────────────────────────────
Validação do alpha via decomposição Fama-French 4 fatores (NEFIN).

HIPÓTESE A SER TESTADA
───────────────────────
H₀: α = 0  (retorno do portfólio é explicado integralmente pelos fatores de risco)
H₁: α > 0  (existe alpha genuíno não capturado por mercado, tamanho, valor, momentum)

Modelo de 4 fatores (versão brasileira via NEFIN/FEA-USP):
  r_p(t) - Rf(t) = α + β₁·(Rm-Rf)(t) + β₂·SMB(t) + β₃·HML(t) + β₄·WML(t) + ε(t)

Critério de aprovação (Sprint 6):
  α com t-stat > 2.0 (equivalente a p-value < 0.025 unilateral)

REFERÊNCIAS
───────────
  Fama & French (1993). Common risk factors in the returns on stocks and bonds.
  Fama & French (2015). A five-factor asset pricing model.
  NEFIN: http://nefin.com.br — fatores calibrados para a B3.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

from src.data.config import get


class FamaFrenchAnalyzer:
    """
    Roda a regressão de 4 fatores NEFIN e extrai o alpha do portfólio.

    Uso típico
    ----------
    ff = FamaFrenchAnalyzer()
    result = ff.run_regression(portfolio_returns, nefin_factors)
    ff.print_results(result)
    passed = ff.check_sprint6_gate(result)
    """

    def __init__(self) -> None:
        cfg = get("validation")
        self.factors        = cfg.get("ff_factors", ["Rm_minus_Rf", "SMB", "HML", "WML"])
        self.min_tstat      = cfg.get("alpha_min_tstat", 2.0)

    # ──────────────────────────────────────────────────────────────
    # REGRESSÃO PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def run_regression(
        self,
        portfolio_returns: pd.Series,
        nefin_factors: pd.DataFrame,
        newey_west_lags: int = 5,
    ) -> dict:
        """
        Executa a regressão OLS de 4 fatores com erros Newey-West.

        Por que Newey-West?
        - Retornos de portfólios podem ter autocorrelação (especialmente
          com rebalanceamento semanal)
        - Newey-West corrige tanto autocorrelação quanto heteroscedasticidade
        - Standard em finanças empíricas

        Parameters
        ----------
        portfolio_returns : retornos líquidos do portfólio (diários)
        nefin_factors     : fatores NEFIN com colunas [Rm_minus_Rf, SMB, HML, WML, Rf]
        newey_west_lags   : número de lags para Newey-West (5 ≈ 1 semana)

        Returns
        -------
        dict com alpha, betas, t-stats, p-values, R², etc.
        """
        # Alinhar datas
        common = portfolio_returns.index.intersection(nefin_factors.index)
        if len(common) < 60:
            raise ValueError(
                f"Apenas {len(common)} datas em comum. "
                f"Mínimo é 60 para regressão confiável."
            )

        r_p = portfolio_returns.loc[common]
        factors = nefin_factors.loc[common]

        # Excesso de retorno: r_p - Rf
        rf = factors["Rf"] if "Rf" in factors.columns else pd.Series(0, index=common)
        y = r_p - rf

        # Regressores: fatores de risco
        available_factors = [f for f in self.factors if f in factors.columns]
        if len(available_factors) < 3:
            raise ValueError(
                f"Fatores insuficientes no NEFIN. Disponíveis: {list(factors.columns)}"
            )

        X = factors[available_factors]
        X = sm.add_constant(X)  # adiciona intercepto (= alpha)

        # OLS com erros Newey-West
        model = sm.OLS(y, X)
        result_ols = model.fit(cov_type="HAC", cov_kwds={"maxlags": newey_west_lags})

        # Extrair alpha
        alpha_val    = result_ols.params["const"]
        alpha_tstat  = result_ols.tvalues["const"]
        alpha_pvalue = result_ols.pvalues["const"]
        alpha_annual = alpha_val * 252  # anualizar

        # Betas dos fatores
        betas = result_ols.params.drop("const")
        tstats_betas = result_ols.tvalues.drop("const")

        result = {
            "alpha_diario":    alpha_val,
            "alpha_anual":     alpha_annual,
            "alpha_tstat":     alpha_tstat,
            "alpha_pvalue":    alpha_pvalue,
            "alpha_annualized_pct": alpha_annual * 100,
            "betas":           betas.to_dict(),
            "t_stats_betas":   tstats_betas.to_dict(),
            "r_squared":       result_ols.rsquared,
            "r_squared_adj":   result_ols.rsquared_adj,
            "n_obs":           len(y),
            "factors_used":    available_factors,
            "ols_result":      result_ols,  # objeto completo para diagnóstico
        }

        logger.info(
            f"Regressão FF4 concluída:\n"
            f"  α = {alpha_val:.6f}/dia = {alpha_annual:.2%}/ano | "
            f"t-stat = {alpha_tstat:.4f} | p-value = {alpha_pvalue:.4f}\n"
            f"  R² = {result_ols.rsquared:.4f}"
        )
        return result

    def run_rolling_regression(
        self,
        portfolio_returns: pd.Series,
        nefin_factors: pd.DataFrame,
        window: int = 252,
        min_periods: int = 126,
    ) -> pd.DataFrame:
        """
        Regressão rolling para detectar degradação do alpha ao longo do tempo.

        Útil para verificar se o alpha é estável ou concentrado em um sub-período.

        Returns
        -------
        pd.DataFrame com alpha e t-stat por data.
        """
        common = portfolio_returns.index.intersection(nefin_factors.index)
        r_p = portfolio_returns.loc[common]
        factors = nefin_factors.loc[common]

        rf = factors["Rf"] if "Rf" in factors.columns else pd.Series(0, index=common)
        y = (r_p - rf)

        available_factors = [f for f in self.factors if f in factors.columns]
        X = sm.add_constant(factors[available_factors])

        results = []
        dates = y.index

        for i in range(len(dates)):
            if i < min_periods:
                results.append({"date": dates[i], "alpha": np.nan, "t_stat": np.nan})
                continue

            start = max(0, i - window + 1)
            y_w = y.iloc[start:i+1]
            X_w = X.iloc[start:i+1]

            try:
                m = sm.OLS(y_w, X_w).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
                results.append({
                    "date":    dates[i],
                    "alpha":   m.params["const"],
                    "t_stat":  m.tvalues["const"],
                })
            except Exception:
                results.append({"date": dates[i], "alpha": np.nan, "t_stat": np.nan})

        df = pd.DataFrame(results).set_index("date")
        df["alpha_annual"] = df["alpha"] * 252
        return df

    # ──────────────────────────────────────────────────────────────
    # GATE E RELATÓRIO
    # ──────────────────────────────────────────────────────────────
    def check_sprint6_gate(self, result: dict) -> bool:
        """
        Gate de saída do Sprint 6: α com t-stat > 2.0.

        t-stat > 2.0 equivale a p-value < 0.025 (unilateral) ou
        p-value < 0.05 (bilateral) — estatisticamente significativo.
        """
        tstat = result["alpha_tstat"]
        passed = abs(tstat) > self.min_tstat and result["alpha_diario"] > 0

        if passed:
            logger.info(
                f"✅ Sprint 6 Gate: PASSOU\n"
                f"   α = {result['alpha_annualized_pct']:.2f}%/ano | "
                f"t-stat = {tstat:.4f} > {self.min_tstat}"
            )
        else:
            logger.warning(
                f"❌ Sprint 6 Gate: FALHOU\n"
                f"   α = {result['alpha_annualized_pct']:.2f}%/ano | "
                f"t-stat = {tstat:.4f} (mínimo: {self.min_tstat})\n"
                f"   O portfólio não demonstra alpha geométrico estatisticamente significativo."
            )
        return passed

    def print_results(self, result: dict) -> None:
        """Exibe tabela formatada dos resultados da regressão."""
        print("=" * 60)
        print("DECOMPOSIÇÃO FAMA-FRENCH 4 FATORES (NEFIN)")
        print("=" * 60)
        print(f"\nAlpha diário : {result['alpha_diario']:+.6f}")
        print(f"Alpha anual  : {result['alpha_annualized_pct']:+.2f}%")
        print(f"t-stat       : {result['alpha_tstat']:+.4f}")
        print(f"p-value      : {result['alpha_pvalue']:.4f}")
        print(f"R²           : {result['r_squared']:.4f}")
        print(f"N obs        : {result['n_obs']}")
        print(f"\nBetas dos fatores:")
        for f, b in result["betas"].items():
            t = result["t_stats_betas"][f]
            print(f"  {f:20s}: β = {b:+.4f}  (t = {t:+.4f})")
        print("=" * 60)
