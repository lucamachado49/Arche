"""
src/validation/permutation.py
──────────────────────────────
Permutation test para validação da hipótese do GDS.

MOTIVAÇÃO
──────────
Com múltiplos ativos e múltiplos horizontes, há risco de encontrar
um IC > 0.03 puramente por acaso (problema de múltiplas comparações).
O permutation test resolve isso sem fazer suposições sobre a distribuição.

METODOLOGIA
────────────
1. Calcular IC observado: IC_obs = corr(GDS, retorno_futuro)
2. Embaralhar os rótulos temporais do GDS (destruir a ordem temporal)
3. Recalcular IC para cada permutação → distribuição sob H₀
4. p-value = P(IC_permutado >= IC_obs) sob H₀

Se p-value < 0.05, rejeitamos H₀: o IC observado não é fruto do acaso.

LOOK-AHEAD SAFETY
──────────────────
O permutation test usa os MESMOS retornos futuros do IC original.
O embaralhamento é apenas nas DATAS do GDS (não nos retornos).
Isso destrói a estrutura temporal sem introduzir informação futura. ✓
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from joblib import Parallel, delayed

from src.data.config import get


class PermutationTest:
    """
    Permutation test para avaliar a significância estatística do IC do GDS.

    Uso típico
    ----------
    pt = PermutationTest()
    result = pt.run(gds, returns, horizon=5)
    print(f"p-value: {result['p_value']:.4f}")
    """

    def __init__(self) -> None:
        cfg = get("validation")
        self.n_permutations = cfg.get("permutation_n", 1000)

    # ──────────────────────────────────────────────────────────────
    # TESTE PRINCIPAL
    # ──────────────────────────────────────────────────────────────
    def run(
        self,
        gds: pd.DataFrame,
        returns: pd.DataFrame,
        horizon: int = 5,
        n_permutations: int | None = None,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> dict:
        """
        Executa o permutation test para um horizonte específico.

        Parameters
        ----------
        gds           : DataFrame de GDS(t) por ativo
        returns       : DataFrame de retornos diários
        horizon       : horizonte de previsão (dias)
        n_permutations: número de permutações (default: settings.yaml)
        n_jobs        : paralelismo (-1 = todos os cores)
        seed          : semente para reprodutibilidade

        Returns
        -------
        dict com: ic_observed, ic_null_dist, p_value, z_score, passed
        """
        n = n_permutations or self.n_permutations
        rng = np.random.default_rng(seed)

        # IC observado
        ic_obs = self._compute_mean_ic(gds, returns, horizon)
        logger.info(f"IC observado (h={horizon}): {ic_obs:.4f}")

        # Distribuição nula via permutação
        logger.info(f"Gerando {n} permutações...")

        if n_jobs != 1:
            null_ics = Parallel(n_jobs=n_jobs)(
                delayed(self._permuted_ic)(gds, returns, horizon, rng.integers(0, 2**31))
                for _ in range(n)
            )
        else:
            null_ics = [
                self._permuted_ic(gds, returns, horizon, int(rng.integers(0, 2**31)))
                for _ in range(n)
            ]

        null_ics = np.array([x for x in null_ics if not np.isnan(x)])

        if len(null_ics) < 10:
            logger.error("Permutações insuficientes para calcular p-value.")
            return {"p_value": np.nan, "ic_observed": ic_obs, "passed": False}

        # p-value bilateral (usamos |IC| para não depender do sinal)
        p_value = (np.abs(null_ics) >= abs(ic_obs)).mean()

        # z-score em relação à distribuição nula
        null_mean = null_ics.mean()
        null_std  = null_ics.std()
        z_score   = (ic_obs - null_mean) / null_std if null_std > 1e-9 else 0.0

        passed = p_value < 0.05

        result = {
            "horizon":        horizon,
            "ic_observed":    round(ic_obs, 4),
            "ic_null_mean":   round(null_mean, 4),
            "ic_null_std":    round(null_std, 4),
            "ic_null_dist":   null_ics,
            "p_value":        round(p_value, 4),
            "z_score":        round(z_score, 4),
            "n_permutations": len(null_ics),
            "passed":         passed,
        }

        status = "✅" if passed else "❌"
        logger.info(
            f"{status} Permutation test (h={horizon}): "
            f"IC={ic_obs:.4f}, p={p_value:.4f}, z={z_score:.4f}"
        )
        return result

    def run_all_horizons(
        self,
        gds: pd.DataFrame,
        returns: pd.DataFrame,
        horizons: list[int] | None = None,
        n_permutations: int | None = None,
    ) -> pd.DataFrame:
        """
        Roda o permutation test para todos os horizontes.

        Returns
        -------
        pd.DataFrame com resultados por horizonte.
        """
        horizons = horizons or [1, 5, 10, 21]
        rows = []
        for h in horizons:
            result = self.run(gds, returns, horizon=h, n_permutations=n_permutations)
            rows.append({
                "horizonte":       h,
                "ic_observado":    result["ic_observed"],
                "ic_null_media":   result["ic_null_mean"],
                "ic_null_std":     result["ic_null_std"],
                "p_value":         result["p_value"],
                "z_score":         result["z_score"],
                "n_permutacoes":   result["n_permutations"],
                "significativo":   result["passed"],
            })

        df = pd.DataFrame(rows).set_index("horizonte")
        return df

    # ──────────────────────────────────────────────────────────────
    # AUXILIARES
    # ──────────────────────────────────────────────────────────────
    def _compute_mean_ic(
        self,
        gds: pd.DataFrame,
        returns: pd.DataFrame,
        horizon: int,
    ) -> float:
        """
        Calcula IC médio cross-seccional para um horizonte.

        IC(t) = Spearman(GDS_i(t), r_i(t+1 → t+h)) para todo i.
        IC_médio = média temporal de IC(t).
        """
        ic_values = []
        for date in gds.index:
            gds_t = gds.loc[date].dropna()
            future = self._get_forward_return(returns, date, horizon)
            if future is None:
                continue

            common = gds_t.index.intersection(future.index)
            if len(common) < 5:
                continue

            x = gds_t.loc[common].values
            y = future.loc[common].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue

            ic, _ = stats.spearmanr(x[mask], y[mask])
            if not np.isnan(ic):
                ic_values.append(ic)

        return float(np.mean(ic_values)) if ic_values else np.nan

    def _permuted_ic(
        self,
        gds: pd.DataFrame,
        returns: pd.DataFrame,
        horizon: int,
        seed: int,
    ) -> float:
        """
        Calcula IC com GDS embaralhado temporalmente.
        Destrói a estrutura temporal do sinal sem alterar os retornos.
        """
        rng = np.random.default_rng(seed)
        # Embaralhar as datas do GDS (shuffle das linhas)
        shuffled_index = rng.permutation(len(gds))
        gds_permuted = gds.iloc[shuffled_index].copy()
        gds_permuted.index = gds.index  # manter o index original
        return self._compute_mean_ic(gds_permuted, returns, horizon)

    def _get_forward_return(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        horizon: int,
    ) -> pd.Series | None:
        """Retorno acumulado de t+1 até t+h."""
        future_dates = returns.index[returns.index > date]
        if len(future_dates) < horizon:
            return None
        return returns.loc[future_dates[:horizon]].sum()
