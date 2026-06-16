"""
src/transfer/pooled_test.py
───────────────────────────
Testes de hipótese POOLED cross-market para a lei GDS→retorno.

O que se testa (H₀):
    "O GDS não tem relação preditiva com o retorno futuro em NENHUM mercado."

Três testes complementares, do mais ao menos rigoroso para a banca:

  1. cross_market_replication()  — pré-registro: calibra h no mercado de
     descoberta (S&P, alto poder), CONGELA, e confirma uma vez no Ibovespa.
     Sem tuning no Brasil → impossível overfittar a amostra pequena.

  2. pooled_ic_meta_analysis()   — meta-análise de Fisher-z ponderada por
     (n−3). O S&P (500 nomes) contribui ~6× o peso do Ibov por data,
     apertando o IC da lei combinada.

  3. stratified_permutation_test() — extensão do permutation test do projeto:
     permuta o GDS DENTRO de cada mercado e pooled. Assumption-light.

  4. conditional_event_test() — teste DIRETO da hipótese do README:
         E[r(t+1:t+h) | GDS < −z·σ]  >  E[r(t+1:t+h)] + α
     Mede o retorno forward médio dos ativos em evento de colapso de
     curvatura vs a média cross-seccional da data, pooled entre mercados,
     com inferência por permutação estratificada (one-sided, direcional).

LOOK-AHEAD SAFETY
─────────────────
O IC usa retorno forward r(t+1:t+h), calculado SÓ na análise ex-post.
O GDS(t) é o preditor; nunca há cruzamento temporal. O embaralhamento
das permutações afeta apenas as DATAS do GDS, nunca os retornos. ✓
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from joblib import Parallel, delayed

from src.data.config import get


# ──────────────────────────────────────────────────────────────────
# HELPERS DE IC CROSS-SECCIONAL (consistentes com src/signal/ic.py)
# ──────────────────────────────────────────────────────────────────
def _forward_return(
    returns: pd.DataFrame,
    date: pd.Timestamp,
    horizon: int,
) -> Optional[pd.Series]:
    """Retorno log acumulado de t+1 até t+h (mesma convenção do projeto)."""
    after = returns.index[returns.index > date]
    if len(after) < horizon:
        return None
    return returns.loc[after[:horizon]].sum().dropna()


def _spearman_ic(g: pd.Series, fwd: pd.Series, min_names: int) -> Optional[tuple[float, int]]:
    """IC de Spearman entre GDS e retorno forward sobre os ativos em comum."""
    common = g.index.intersection(fwd.index)
    if len(common) < min_names:
        return None
    x = g.loc[common].to_numpy()
    y = fwd.loc[common].to_numpy()
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < min_names:
        return None
    ic, _ = stats.spearmanr(x[m], y[m])
    if np.isnan(ic):
        return None
    return float(ic), int(m.sum())


class _MarketICCache:
    """
    Pré-computa, por mercado, o retorno forward de cada data UMA vez.

    Sob permutação, só a linha de GDS atribuída a cada data muda — o
    retorno forward é fixo. Cachear isso torna o permutation test viável
    mesmo com 500 nomes × centenas de datas × 1000 permutações.
    """

    def __init__(
        self,
        gds: pd.DataFrame,
        returns: pd.DataFrame,
        horizon: int,
        min_names: int,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.min_names = min_names
        idx = dates if dates is not None else gds.index
        self.gds_rows: list[pd.Series] = []
        self.fwd_rows: list[Optional[pd.Series]] = []
        for d in idx:
            self.gds_rows.append(gds.loc[d].dropna())
            self.fwd_rows.append(_forward_return(returns, d, horizon))
        self.n_dates = len(idx)

    def ic_panel(self, perm: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula (ic, n) por data. perm[pos] = índice da linha de GDS
        atribuída à posição pos. perm=None → atribuição identidade (observado).
        """
        ics, ns = [], []
        for pos in range(self.n_dates):
            fwd = self.fwd_rows[pos]
            if fwd is None:
                continue
            src = pos if perm is None else perm[pos]
            res = _spearman_ic(self.gds_rows[src], fwd, self.min_names)
            if res is not None:
                ics.append(res[0])
                ns.append(res[1])
        return np.asarray(ics), np.asarray(ns, dtype=float)

    def event_panel(
        self,
        threshold_z: float,
        min_events: int = 2,
        perm: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Excesso de retorno condicional por data:

            stat(t) = mean(fwd[GDS < −z]) − mean(fwd[todos válidos])

        Demean cross-seccional remove o movimento de mercado da data —
        o que sobra é o excesso atribuível ao evento de colapso de curvatura.
        Retorna (stats, n_events) sobre as datas com ≥ min_events eventos.
        """
        stats_, n_events_ = [], []
        for pos in range(self.n_dates):
            fwd = self.fwd_rows[pos]
            if fwd is None:
                continue
            src = pos if perm is None else perm[pos]
            g = self.gds_rows[src]
            common = g.index.intersection(fwd.index)
            if len(common) < self.min_names:
                continue
            x = g.loc[common].to_numpy()
            y = fwd.loc[common].to_numpy()
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() < self.min_names:
                continue
            x, y = x[m], y[m]
            events = x < -threshold_z
            if events.sum() < min_events:
                continue
            stats_.append(float(y[events].mean() - y.mean()))
            n_events_.append(int(events.sum()))
        return np.asarray(stats_), np.asarray(n_events_, dtype=float)


def _subsample_nonoverlapping(
    gds_index: pd.DatetimeIndex,
    returns_index: pd.DatetimeIndex,
    horizon: int,
) -> pd.DatetimeIndex:
    """
    Seleciona datas cujas janelas forward de h dias úteis NÃO se sobrepõem.

    Mede a distância em dias ÚTEIS via posição no índice diário de retornos,
    garantindo independência aproximada entre ICs consecutivos.
    """
    pos = returns_index.searchsorted(pd.DatetimeIndex(gds_index))
    keep, last = [], -10**9
    for d, p in sorted(zip(gds_index, pos), key=lambda t: t[1]):
        if p - last >= horizon:
            keep.append(d)
            last = p
    return pd.DatetimeIndex(keep)


def _weighted_mean_ic(ics: np.ndarray, ns: np.ndarray, weight_by_n: bool) -> float:
    """Média de IC ponderada por nº de ativos (precisão cross-seccional)."""
    if len(ics) == 0:
        return np.nan
    w = ns if weight_by_n else np.ones_like(ns)
    return float(np.sum(w * ics) / np.sum(w))


def _newey_west_tstat(x: np.ndarray, lags: int) -> float:
    """t-stat da média de x com erro-padrão HAC (Newey-West)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        cov = (e[k:] @ e[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return float(mu / se) if se > 0 else np.nan


# ──────────────────────────────────────────────────────────────────
# TESTE PRINCIPAL
# ──────────────────────────────────────────────────────────────────
class CrossMarketHypothesisTest:
    """
    Bateria de testes pooled para a hipótese do GDS usando múltiplos mercados.

    Uso típico
    ----------
    test = CrossMarketHypothesisTest()
    panels = {"BR": (gds_br, ret_br), "US": (gds_us, ret_us)}

    # 1. Replicação (o mais defensável)
    rep = test.cross_market_replication(panels)

    # 2. Meta-análise de IC
    meta = test.pooled_ic_meta_analysis(panels, horizon=5)

    # 3. Permutation test estratificado
    perm = test.stratified_permutation_test(panels, horizon=5)
    """

    def __init__(self) -> None:
        cfg = get("transfer") or {}
        self.horizons = cfg.get("ic_horizons", [1, 5, 10, 21])
        self.n_permutations = cfg.get("permutation_n", 1000)
        self.min_names = cfg.get("min_names_per_date", 5)
        self.non_overlapping = cfg.get("non_overlapping", True)
        self.discovery = cfg.get("discovery_market", "US")
        self.confirm = cfg.get("confirm_market", "BR")

    # ──────────────────────────────────────────────────────────────
    # 1. REPLICAÇÃO CROSS-MARKET (pré-registro)
    # ──────────────────────────────────────────────────────────────
    def cross_market_replication(
        self,
        panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        discovery: Optional[str] = None,
        confirm: Optional[str] = None,
        horizons: Optional[list[int]] = None,
    ) -> dict:
        """
        Calibra o horizonte no mercado de descoberta e confirma no outro.

        Protocolo (evita data snooping na amostra pequena):
          1. No mercado de DESCOBERTA (S&P), escolhe h* que maximiza |IC_IR|.
          2. CONGELA h*.
          3. No mercado de CONFIRMAÇÃO (Ibov), reporta IC, IC_IR, t-stat (NW)
             e p-value de permutação SÓ em h* — um único teste, sem tuning.

        Returns
        -------
        dict com h* escolhido, métricas na descoberta e na confirmação.
        """
        discovery = discovery or self.discovery
        confirm = confirm or self.confirm
        horizons = horizons or self.horizons

        if discovery not in panels or confirm not in panels:
            raise ValueError(
                f"Mercados '{discovery}'/'{confirm}' ausentes em panels={list(panels)}"
            )

        gds_d, ret_d = panels[discovery]
        gds_c, ret_c = panels[confirm]

        # ── Passo 1-2: descobrir h* no mercado de descoberta ──
        disc_metrics = {}
        for h in horizons:
            disc_metrics[h] = self._single_market_ic_stats(gds_d, ret_d, h)
        h_star = max(horizons, key=lambda h: abs(disc_metrics[h]["ic_ir"]))

        logger.info(
            f"[Replicação] Descoberta ({discovery}): h*={h_star} "
            f"(IC={disc_metrics[h_star]['ic_mean']:.4f}, "
            f"IC_IR={disc_metrics[h_star]['ic_ir']:.4f})"
        )

        # ── Passo 3: confirmar em h* no mercado de confirmação ──
        conf = self._single_market_ic_stats(gds_c, ret_c, h_star)
        conf_perm = self.stratified_permutation_test(
            {confirm: panels[confirm]}, horizon=h_star
        )

        passed = (
            np.sign(conf["ic_mean"]) == np.sign(disc_metrics[h_star]["ic_mean"])
            and conf_perm["p_value"] < 0.05
        )

        result = {
            "discovery_market": discovery,
            "confirm_market": confirm,
            "h_star": h_star,
            "discovery": disc_metrics[h_star],
            "confirmation": conf,
            "confirmation_perm_p": conf_perm["p_value"],
            "same_sign": bool(
                np.sign(conf["ic_mean"]) == np.sign(disc_metrics[h_star]["ic_mean"])
            ),
            "passed": bool(passed),
        }

        status = "✅" if passed else "❌"
        logger.info(
            f"{status} [Replicação] Confirmação ({confirm}) em h*={h_star}: "
            f"IC={conf['ic_mean']:.4f}, t_NW={conf['t_nw']:.2f}, "
            f"p_perm={conf_perm['p_value']:.4f}"
        )
        return result

    # ──────────────────────────────────────────────────────────────
    # 2. META-ANÁLISE DE IC (Fisher-z, inverso-variância)
    # ──────────────────────────────────────────────────────────────
    def pooled_ic_meta_analysis(
        self,
        panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        horizon: int = 5,
    ) -> dict:
        """
        Combina os ICs cross-seccionais de todos os mercados via Fisher-z.

        Para cada (mercado, data): r = IC de Spearman, n = nº de ativos.
        Transformação de Fisher z = arctanh(r), com Var(z) ≈ 1/(n−3).
        Pooled por inverso-variância (peso = n−3):

            z_pool = Σ w·z / Σ w ,  SE = √(1/Σ w) ,  Z = z_pool/SE
            IC_pool = tanh(z_pool)

        Mercados com cross-section maior (S&P) recebem mais peso — é assim
        que "mais dados" aperta o teste, sem misturar geometrias.

        Datas são subamostradas para não-sobreposição (se configurado),
        garantindo independência aproximada entre ICs.
        """
        all_z, all_w = [], []
        per_market = {}

        for market, (gds, returns) in panels.items():
            dates = gds.index
            if self.non_overlapping:
                dates = _subsample_nonoverlapping(dates, returns.index, horizon)

            cache = _MarketICCache(gds, returns, horizon, self.min_names, dates=dates)
            ics, ns = cache.ic_panel()

            valid = ns > 3
            if valid.sum() == 0:
                logger.warning(f"[{market}] sem ICs válidos (n>3) em h={horizon}.")
                continue

            r = np.clip(ics[valid], -0.999, 0.999)
            z = np.arctanh(r)
            w = ns[valid] - 3.0

            all_z.append(z)
            all_w.append(w)

            zp = np.sum(w * z) / np.sum(w)
            se = np.sqrt(1.0 / np.sum(w))
            per_market[market] = {
                "ic_pooled": float(np.tanh(zp)),
                "z_stat": float(zp / se),
                "p_value": float(2 * stats.norm.sf(abs(zp / se))),
                "n_dates": int(valid.sum()),
                "mean_n_assets": float(np.mean(ns[valid])),
            }

        if not all_z:
            return {"passed": False, "p_value": np.nan, "horizon": horizon}

        z = np.concatenate(all_z)
        w = np.concatenate(all_w)
        z_pool = np.sum(w * z) / np.sum(w)
        se_pool = np.sqrt(1.0 / np.sum(w))
        Z = z_pool / se_pool
        p_value = float(2 * stats.norm.sf(abs(Z)))

        result = {
            "horizon": horizon,
            "ic_pooled": float(np.tanh(z_pool)),
            "z_stat": float(Z),
            "p_value": p_value,
            "n_observations": int(len(z)),
            "per_market": per_market,
            "passed": bool(p_value < 0.05),
        }

        status = "✅" if result["passed"] else "❌"
        logger.info(
            f"{status} [Meta-análise] h={horizon}: IC_pool={result['ic_pooled']:.4f}, "
            f"Z={Z:.2f}, p={p_value:.4g} ({len(z)} obs de {len(per_market)} mercados)"
        )
        return result

    # ──────────────────────────────────────────────────────────────
    # 3. PERMUTATION TEST ESTRATIFICADO (within-market)
    # ──────────────────────────────────────────────────────────────
    def stratified_permutation_test(
        self,
        panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        horizon: int = 5,
        n_permutations: Optional[int] = None,
        weight_by_n: bool = True,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> dict:
        """
        Permutation test pooled com estratificação por mercado.

        Estatística observada: média de IC ponderada por nº de ativos,
        sobre TODAS as datas de TODOS os mercados.

        Distribuição nula: embaralha as linhas de GDS DENTRO de cada
        mercado independentemente (preserva a estrutura cross-seccional e
        a identidade de mercado sob H₀), recalcula a estatística pooled.

        p-value bilateral = P(|IC_perm| ≥ |IC_obs|) sob H₀.
        """
        n_perm = n_permutations or self.n_permutations
        rng = np.random.default_rng(seed)

        # Pré-computar caches por mercado (forward returns fixos)
        caches = self._build_caches(panels, horizon)

        def pooled_stat(perms: dict[str, Optional[np.ndarray]]) -> float:
            ics_all, ns_all = [], []
            for market, cache in caches.items():
                ics, ns = cache.ic_panel(perms.get(market))
                ics_all.append(ics)
                ns_all.append(ns)
            ics_all = np.concatenate(ics_all) if ics_all else np.array([])
            ns_all = np.concatenate(ns_all) if ns_all else np.array([])
            return _weighted_mean_ic(ics_all, ns_all, weight_by_n)

        # Observado (atribuição identidade)
        ic_obs = pooled_stat({m: None for m in caches})

        # Distribuição nula
        seeds = rng.integers(0, 2**31, size=n_perm)

        def one_perm(s: int) -> float:
            r = np.random.default_rng(s)
            perms = {m: r.permutation(c.n_dates) for m, c in caches.items()}
            return pooled_stat(perms)

        if n_jobs != 1:
            null = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(one_perm)(int(s)) for s in seeds
            )
        else:
            null = [one_perm(int(s)) for s in seeds]

        null = np.array([x for x in null if not np.isnan(x)])
        if len(null) < 10:
            return {"passed": False, "p_value": np.nan, "ic_observed": ic_obs}

        p_value = float((np.abs(null) >= abs(ic_obs)).mean())
        z_score = float((ic_obs - null.mean()) / null.std()) if null.std() > 1e-12 else 0.0

        result = {
            "horizon": horizon,
            "ic_observed": float(ic_obs),
            "ic_null_mean": float(null.mean()),
            "ic_null_std": float(null.std()),
            "p_value": p_value,
            "z_score": z_score,
            "n_permutations": int(len(null)),
            "markets": list(caches.keys()),
            "passed": bool(p_value < 0.05),
        }

        status = "✅" if result["passed"] else "❌"
        logger.info(
            f"{status} [Permutação estratificada] h={horizon}: "
            f"IC_obs={ic_obs:.4f}, p={p_value:.4f}, z={z_score:.2f} "
            f"({'+'.join(caches.keys())})"
        )
        return result

    # ──────────────────────────────────────────────────────────────
    # 4. TESTE CONDICIONAL DE EVENTO (forma direta do README)
    # ──────────────────────────────────────────────────────────────
    def conditional_event_test(
        self,
        panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        horizon: int = 5,
        threshold_z: Optional[float] = None,
        min_events: int = 2,
        n_permutations: Optional[int] = None,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> dict:
        """
        Teste direto da hipótese central do projeto:

            E[r_i(t+1:t+h) | GDS_i(t) < −z·σ]  >  E[r_i(t+1:t+h)] + α,  α > 0

        Estatística observada: média (sobre datas e mercados) do excesso de
        retorno dos ativos em evento vs a média cross-seccional da data.

        Inferência: permutação estratificada within-market das linhas de GDS.
        Por ser uma hipótese DIRECIONAL (α > 0), o p-value principal é
        one-sided: P(stat_perm ≥ stat_obs). O two-sided é reportado junto.
        """
        z = threshold_z if threshold_z is not None else get(
            "curvature", "gds_threshold_sigma", 2.0
        )
        n_perm = n_permutations or self.n_permutations
        rng = np.random.default_rng(seed)

        caches = self._build_caches(panels, horizon)

        def pooled_event_stat(perms: dict[str, Optional[np.ndarray]]) -> tuple[float, float, int]:
            stats_all, nev_all = [], []
            for market, cache in caches.items():
                s, ne = cache.event_panel(z, min_events, perms.get(market))
                stats_all.append(s)
                nev_all.append(ne)
            stats_all = np.concatenate(stats_all) if stats_all else np.array([])
            nev_all = np.concatenate(nev_all) if nev_all else np.array([])
            if len(stats_all) == 0:
                return np.nan, 0.0, 0
            return float(stats_all.mean()), float(nev_all.mean()), int(len(stats_all))

        alpha_obs, mean_events, n_event_dates = pooled_event_stat({m: None for m in caches})

        if n_event_dates < 10 or np.isnan(alpha_obs):
            logger.warning(
                f"[Evento condicional] h={horizon}, z={z}: apenas {n_event_dates} "
                f"datas com ≥{min_events} eventos — sem poder estatístico. "
                f"Considere reduzir threshold_z."
            )
            return {
                "horizon": horizon, "threshold_z": z, "alpha_obs": alpha_obs,
                "n_event_dates": n_event_dates, "p_one_sided": np.nan,
                "p_two_sided": np.nan, "passed": False,
            }

        seeds = rng.integers(0, 2**31, size=n_perm)

        def one_perm(s: int) -> float:
            r = np.random.default_rng(s)
            perms = {m: r.permutation(c.n_dates) for m, c in caches.items()}
            return pooled_event_stat(perms)[0]

        if n_jobs != 1:
            null = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(one_perm)(int(s)) for s in seeds
            )
        else:
            null = [one_perm(int(s)) for s in seeds]

        null = np.array([x for x in null if not np.isnan(x)])
        p_one = float((null >= alpha_obs).mean())
        p_two = float((np.abs(null) >= abs(alpha_obs)).mean())

        result = {
            "horizon": horizon,
            "threshold_z": float(z),
            "alpha_obs": float(alpha_obs),
            "alpha_obs_bps": float(alpha_obs * 1e4),
            "mean_events_per_date": float(mean_events),
            "n_event_dates": int(n_event_dates),
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "p_one_sided": p_one,
            "p_two_sided": p_two,
            "n_permutations": int(len(null)),
            "markets": list(caches.keys()),
            "passed": bool(p_one < 0.05 and alpha_obs > 0),
        }

        status = "✅" if result["passed"] else "❌"
        logger.info(
            f"{status} [Evento condicional] h={horizon}, z={z}: "
            f"α={alpha_obs*1e4:+.1f} bps/{horizon}d sobre {n_event_dates} datas "
            f"(p_1s={p_one:.4f}, p_2s={p_two:.4f})"
        )
        return result

    # ──────────────────────────────────────────────────────────────
    # AUXILIAR: caches por mercado (forward returns pré-computados)
    # ──────────────────────────────────────────────────────────────
    def _build_caches(
        self,
        panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        horizon: int,
    ) -> dict[str, _MarketICCache]:
        caches = {}
        for market, (gds, returns) in panels.items():
            dates = gds.index
            if self.non_overlapping:
                dates = _subsample_nonoverlapping(dates, returns.index, horizon)
            caches[market] = _MarketICCache(
                gds, returns, horizon, self.min_names, dates=dates
            )
        return caches

    # ──────────────────────────────────────────────────────────────
    # AUXILIAR: estatísticas de IC de um único mercado
    # ──────────────────────────────────────────────────────────────
    def _single_market_ic_stats(
        self,
        gds: pd.DataFrame,
        returns: pd.DataFrame,
        horizon: int,
    ) -> dict:
        """IC médio, IC_IR e t-stat Newey-West de um mercado em um horizonte."""
        dates = gds.index
        if self.non_overlapping:
            dates = _subsample_nonoverlapping(dates, returns.index, horizon)
        cache = _MarketICCache(gds, returns, horizon, self.min_names, dates=dates)
        ics, _ = cache.ic_panel()

        if len(ics) < 3:
            return {"ic_mean": np.nan, "ic_ir": np.nan, "t_nw": np.nan, "n_dates": len(ics)}

        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics, ddof=1))
        ic_ir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
        # lag NW ≈ horizonte (resíduos sobrepostos remanescentes)
        t_nw = _newey_west_tstat(ics, lags=max(1, horizon))

        return {
            "ic_mean": ic_mean,
            "ic_ir": float(ic_ir),
            "t_nw": float(t_nw),
            "n_dates": int(len(ics)),
        }

    # ──────────────────────────────────────────────────────────────
    # ORQUESTRADOR: roda tudo e devolve um sumário
    # ──────────────────────────────────────────────────────────────
    def run_full_battery(
        self,
        panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        horizons: Optional[list[int]] = None,
    ) -> dict:
        """Roda os três testes em todos os horizontes e devolve um dict-resumo."""
        horizons = horizons or self.horizons
        out = {"replication": self.cross_market_replication(panels), "by_horizon": {}}
        for h in horizons:
            out["by_horizon"][h] = {
                "meta_analysis": self.pooled_ic_meta_analysis(panels, horizon=h),
                "permutation": self.stratified_permutation_test(panels, horizon=h),
                "conditional_event": self.conditional_event_test(panels, horizon=h),
            }
        return out
