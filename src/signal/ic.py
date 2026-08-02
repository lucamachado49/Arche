"""
src/signal/ic.py
────────────────
Information Coefficient (IC) — métrica de qualidade do sinal GDS.

DEFINIÇÃO
──────────
IC(t) = Spearman_corr(GDS_i(t), r_i(t → t+h))

onde:
  GDS_i(t) : Geometric Divergence Score do ativo i na data t
  r_i(t→t+h): retorno acumulado de i entre t+1 e t+h

IC ∈ [-1, 1]:
  IC > 0  : GDS alto prevê retorno alto (relação medida em BR h=21).
            O LADO do book é definido por signal.signal_sign no settings.yaml
            (alpha_score = signal_sign · GDS), aplicado em PortfolioConstructor.
            Com IC > 0 ⇒ signal_sign=+1 (long no GDS alto). Não há inversão
            implícita escondida no pipeline — a convenção é única e explícita.
  |IC| > 0.05 : sinal economicamente significativo
  IC_IR > 0.5  : sinal consistente ao longo do tempo

CRITÉRIOS DE APROVAÇÃO (Sprint 4)
───────────────────────────────────
  - IC médio > |0.03| em pelo menos 2 horizontes
  - IC_IR (IC / std(IC)) > 0.30
  - % períodos com IC > 0 (ou < 0 para sinal invertido) > 55%

LOOK-AHEAD SAFETY
──────────────────
O retorno futuro r_i(t→t+h) é calculado APENAS durante a análise histórica.
No backtest real, o IC é calculado ex-post sobre um período de teste.
Nunca é usado no loop de decisão — apenas para validação do sinal. ✓
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.data.config import get


class ICAnalyzer:
    """
    Calcula e analisa o Information Coefficient do sinal GDS.

    Uso típico
    ----------
    analyzer = ICAnalyzer()
    ic_series = analyzer.compute_ic_series(gds, returns, horizon=5)
    summary = analyzer.summarize(ic_series)
    passed = analyzer.check_gates(summary)
    """

    def __init__(self) -> None:
        cfg = get("signal")
        self.horizons           = cfg.get("ic_horizons", [1, 5, 10, 21])
        self.ic_min_mean        = cfg.get("ic_min_mean", 0.03)
        self.ic_min_ir          = cfg.get("ic_min_ir", 0.30)
        self.ic_min_pct_positive = cfg.get("ic_min_pct_positive", 0.55)

    # ──────────────────────────────────────────────────────────────
    # CÁLCULO DO IC
    # ──────────────────────────────────────────────────────────────
    def compute_ic_series(
        self,
        gds: pd.DataFrame,
        returns: pd.DataFrame,
        horizon: int = 5,
        method: str = "spearman",
    ) -> pd.Series:
        """
        Calcula a série temporal de IC para um dado horizonte.

        LOOK-AHEAD SAFETY:
        O retorno forward é r_i(t+1 → t+h). O GDS em t é o preditor.
        Na análise de IC, usamos os retornos futuros APENAS para medir
        a qualidade ex-post do sinal, nunca para gerar decisões. ✓

        Parameters
        ----------
        gds     : DataFrame de GDS(t) [index=Date, cols=tickers]
        returns : DataFrame de retornos diários [index=Date, cols=tickers]
        horizon : horizonte de previsão em dias úteis
        method  : "spearman" (rank-based, robusto) ou "pearson"

        Returns
        -------
        pd.Series com IC(t) para cada data disponível.
        """
        ic_values = {}

        for date in gds.index:
            gds_t = gds.loc[date].dropna()
            if len(gds_t) < 5:
                continue

            # Retorno cumulativo de t+1 até t+h
            future_return = self._get_forward_return(returns, date, horizon)
            if future_return is None or len(future_return) < 5:
                continue

            # Interseção de tickers disponíveis
            common = gds_t.index.intersection(future_return.index)
            if len(common) < 5:
                continue

            x = gds_t.loc[common].values
            y = future_return.loc[common].values

            # Remover NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue

            if method == "spearman":
                ic, _ = stats.spearmanr(x[mask], y[mask])
            else:
                ic, _ = stats.pearsonr(x[mask], y[mask])

            ic_values[date] = ic

        if not ic_values:
            logger.warning(f"Nenhum IC calculado para horizonte h={horizon}.")
            return pd.Series(dtype=float, name=f"IC_h{horizon}")

        return pd.Series(ic_values, name=f"IC_h{horizon}")

    def compute_all_horizons(
        self,
        gds: pd.DataFrame,
        returns: pd.DataFrame,
        horizons: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Calcula IC para todos os horizontes configurados.

        Returns
        -------
        pd.DataFrame com colunas IC_h1, IC_h5, IC_h10, IC_h21.
        """
        horizons = horizons or self.horizons
        ic_dict = {}

        for h in horizons:
            logger.info(f"Calculando IC para h={h} dias...")
            ic_series = self.compute_ic_series(gds, returns, horizon=h)
            ic_dict[f"IC_h{h}"] = ic_series

        return pd.DataFrame(ic_dict)

    # ──────────────────────────────────────────────────────────────
    # ANÁLISE E GATES
    # ──────────────────────────────────────────────────────────────
    def summarize(self, ic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera resumo estatístico do IC por horizonte.

        Returns
        -------
        pd.DataFrame com linhas = horizontes, colunas = métricas:
          mean, std, IC_IR, t_stat, p_value, pct_positive, pct_negative
        """
        rows = []
        for col in ic_df.columns:
            ic = ic_df[col].dropna()
            if len(ic) < 3:
                continue

            mean   = ic.mean()
            std    = ic.std()
            ic_ir  = mean / std if std > 1e-8 else 0.0
            t_stat = mean / (std / np.sqrt(len(ic))) if std > 1e-8 else 0.0
            p_val  = 2 * stats.t.sf(abs(t_stat), df=len(ic)-1)

            rows.append({
                "horizonte":    col,
                "n_obs":        len(ic),
                "mean":         round(mean, 4),
                "std":          round(std, 4),
                "IC_IR":        round(ic_ir, 4),
                "t_stat":       round(t_stat, 4),
                "p_value":      round(p_val, 4),
                "pct_positive": round((ic > 0).mean(), 4),
                "pct_negative": round((ic < 0).mean(), 4),
                "sharpe_proxy": round(ic_ir * np.sqrt(52), 4),  # anualizado se semanal
            })

        df = pd.DataFrame(rows).set_index("horizonte")
        return df

    def check_gates(self, summary: pd.DataFrame) -> dict[str, bool]:
        """
        Verifica os critérios de aprovação do Sprint 4.

        Gates (pré-definidos antes de ver os dados — evita data snooping):
          1. |IC médio| > ic_min_mean em pelo menos 2 horizontes
          2. |IC_IR| > ic_min_ir em pelo menos 1 horizonte
          3. % períodos no lado certo > ic_min_pct_positive em pelo menos 1 horizonte

        Returns
        -------
        dict com status de cada gate e resultado global.
        """
        gates = {}

        # Gate 1: |IC médio| > threshold em ≥ 2 horizontes
        ic_means = summary["mean"].abs()
        n_passing_mean = (ic_means > self.ic_min_mean).sum()
        gates["ic_mean_2horizons"] = int(n_passing_mean) >= 2
        gates["ic_mean_passing_horizons"] = int(n_passing_mean)

        # Gate 2: |IC_IR| > threshold
        ic_ir = summary["IC_IR"].abs()
        gates["ic_ir_ok"] = bool((ic_ir > self.ic_min_ir).any())
        gates["ic_ir_max"] = round(float(ic_ir.max()), 4)

        # Gate 3: % no lado certo
        # O sinal GDS negativo → long. IC esperado: correlação negativa
        # (GDS negativo, retorno positivo). Mas por convenção, reportamos
        # |IC| e verificamos consistência.
        ic_sign_consistent = summary[["pct_positive", "pct_negative"]].max(axis=1)
        gates["ic_consistency_ok"] = bool((ic_sign_consistent > self.ic_min_pct_positive).any())
        gates["ic_consistency_max"] = round(float(ic_sign_consistent.max()), 4)

        # Resultado global
        gates["all_gates_passed"] = all([
            gates["ic_mean_2horizons"],
            gates["ic_ir_ok"],
            gates["ic_consistency_ok"],
        ])

        status = "✅ PASSOU" if gates["all_gates_passed"] else "❌ FALHOU"
        logger.info(
            f"Gates Sprint 4: {status}\n"
            f"  IC médio ≥ 2 horizontes: {gates['ic_mean_2horizons']} "
            f"({n_passing_mean}/{len(summary)})\n"
            f"  IC_IR máximo: {gates['ic_ir_max']:.4f} "
            f"(mínimo: {self.ic_min_ir})\n"
            f"  Consistência máxima: {gates['ic_consistency_max']:.1%} "
            f"(mínimo: {self.ic_min_pct_positive:.0%})"
        )
        return gates

    # ──────────────────────────────────────────────────────────────
    # UTILIDADES
    # ──────────────────────────────────────────────────────────────
    def _get_forward_return(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        horizon: int,
    ) -> Optional[pd.Series]:
        """
        Calcula retorno acumulado de t+1 até t+h.

        r(t+1 → t+h) = Σ_{k=1}^{h} log_return_{t+k}
                     = ln(P_{t+h} / P_t)

        LOOK-AHEAD NOTE:
        Este método acessa retornos APÓS date. É usado apenas para
        análise histórica de IC (ex-post). No backtest real, só
        os sinais GDS(t) são usados para rebalancear. ✓
        """
        available_after = returns.index[returns.index > date]
        if len(available_after) < horizon:
            return None

        future_dates = available_after[:horizon]
        future_ret = returns.loc[future_dates].sum()  # soma de log-returns = log-return acumulado
        return future_ret.dropna()

    def compute_cumulative_ic(
        self,
        ic_series: pd.Series,
        expanding: bool = True,
    ) -> pd.Series:
        """
        Calcula IC cumulativo (média expansível) ao longo do tempo.
        Útil para gráfico de degradação do sinal.
        """
        if expanding:
            return ic_series.expanding().mean()
        else:
            return ic_series.cumsum() / (np.arange(len(ic_series)) + 1)
