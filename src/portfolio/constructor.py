"""
src/portfolio/constructor.py
────────────────────────────
Construção do portfólio long-short baseado no GDS.

ESTRATÉGIA
──────────
1. Rankear ativos por GDS em cada data de rebalanceamento
2. Quintil Q1 (GDS mais negativo) → LONG  (hipótese: reversão geométrica)
3. Quintil Q5 (GDS mais positivo) → SHORT (hipótese: sobre-extensão geométrica)
4. Pesos: igual-ponderados dentro de cada perna, sujeito a max_weight
5. Portfólio market-neutral: valor long = valor short

LOOK-AHEAD SAFETY
──────────────────
Os pesos em t são calculados apenas com GDS(t) disponível em t.
O retorno do portfólio é realizado de t+1 em diante. ✓
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data.config import get


class PortfolioConstructor:
    """
    Constrói o portfólio long-short a partir do sinal GDS.

    Uso típico
    ----------
    constructor = PortfolioConstructor()
    weights = constructor.build_weights(gds_t, eligible_tickers)
    # weights: Series com pesos positivos (long) e negativos (short)
    """

    def __init__(self) -> None:
        cfg = get("portfolio")
        self.n_quantiles    = cfg.get("n_quantiles", 5)
        self.max_weight     = cfg.get("max_weight_single_asset", 0.15)
        self.long_short_ratio = cfg.get("long_short_ratio", 1.0)
        # Convenção de sinal centralizada (única fonte de verdade do lado do
        # book). alpha_score = signal_sign · GDS; long no maior score.
        # Ver signal.signal_sign em settings.yaml.
        self.signal_sign    = get("signal", "signal_sign", 1)

        # Método de construção:
        #   "quintile" — long/short nos quintis extremos (legado, descarta o
        #                 miolo do cross-section e concentra nas caudas ruidosas)
        #   "zscore"   — peso ∝ z-score do alpha em TODO o cross-section
        #   "mv"       — mean-variance: w ∝ Σ⁻¹·α (usa a covariância do
        #                 universo; cai p/ "zscore" se Σ indisponível)
        self.method         = cfg.get("construction_method", "mv")
        # Banda de no-trade (em unidades de peso): nomes cujo peso-alvo mudou
        # menos que isso não são negociados — corta turnover de ruído.
        self.no_trade_band  = cfg.get("no_trade_band", 0.01)
        # Hysteresis no ranking do quintil (entra no top entry, sai no top exit).
        self.entry_quantile = cfg.get("entry_quantile", 1.0 / self.n_quantiles)
        self.exit_quantile  = cfg.get("exit_quantile", 1.0 / self.n_quantiles)
        # Janela p/ estimar a covariância (consistente com o grafo)
        self.cov_window      = get("graph", "correlation_window_days", 60)
        self.cov_min_periods = get("graph", "correlation_min_periods", 40)

    # ──────────────────────────────────────────────────────────────
    # CONSTRUÇÃO DE PESOS
    # ──────────────────────────────────────────────────────────────
    def build_weights(
        self,
        gds_t: pd.Series,
        eligible_tickers: Optional[list[str]] = None,
        returns: Optional[pd.DataFrame] = None,
        date: Optional[pd.Timestamp] = None,
        prev_long: Optional[set] = None,
        prev_short: Optional[set] = None,
    ) -> pd.Series:
        """
        Constrói pesos do portfólio para uma data de rebalanceamento.

        Todas as construções operam sobre o alpha_score = signal_sign · GDS
        (maior score ⇒ candidato a LONG). A escolha do lado vive em
        settings.yaml (signal.signal_sign), nunca aqui. O método é definido
        por portfolio.construction_method:

          - "quintile": long/short nos quintis extremos do score (legado).
          - "zscore"  : peso ∝ z-score do score em todo o cross-section —
                         captura o IC inteiro, não só as caudas.
          - "mv"      : mean-variance, w ∝ Σ⁻¹·α, usando a covariância
                         (Ledoit-Wolf) estimada na janela até `date`. Exige
                         `returns` e `date`; sem eles, cai para "zscore".

        O portfólio é market-neutral: Σ w_long = 1, Σ w_short = -ratio.

        Parameters
        ----------
        gds_t           : GDS na data t (Series, index=tickers)
        eligible_tickers : filtro adicional de elegibilidade
        returns         : retornos diários (p/ Σ no método "mv"); look-ahead
                          safe — só são usados dados com índice ≤ date
        date            : data de rebalanceamento (p/ recortar a janela de Σ)

        Returns
        -------
        pd.Series com pesos (soma(positivos) ≈ 1, soma(negativos) ≈ -1)
        """
        # Filtrar para universo elegível
        if eligible_tickers is not None:
            gds_t = gds_t.reindex(eligible_tickers).dropna()
        else:
            gds_t = gds_t.dropna()

        if len(gds_t) < self.n_quantiles * 2:
            logger.warning(
                f"Apenas {len(gds_t)} ativos disponíveis. "
                f"Mínimo para {self.n_quantiles} quintis: {self.n_quantiles * 2}."
            )
            return pd.Series(dtype=float)

        # alpha_score: convenção de sinal centralizada
        alpha_score = self.signal_sign * gds_t

        if self.method == "quintile":
            weights = self._quintile_weights(
                alpha_score, gds_t.index, prev_long, prev_short
            )
        else:
            cov = None
            if self.method == "mv" and returns is not None and date is not None:
                cov = self._estimate_cov(returns, date, list(alpha_score.index))
            weights = self._continuous_weights(alpha_score, gds_t.index, cov)

        if weights.empty:
            return weights

        # Cap de peso máximo + renormalização das pernas
        weights = self._apply_weight_cap(weights)
        weights = self._renormalize(weights)
        return weights

    # ──────────────────────────────────────────────────────────────
    # MÉTODOS DE CONSTRUÇÃO
    # ──────────────────────────────────────────────────────────────
    def _quintile_weights(
        self,
        alpha_score: pd.Series,
        full_index: pd.Index,
        prev_long: Optional[set] = None,
        prev_short: Optional[set] = None,
    ) -> pd.Series:
        """
        Long no top do score, short na base (igual-ponderado dentro da perna).

        Com hysteresis (exit_quantile > entry_quantile): um nome ENTRA na perna
        se está na cauda de entrada (top/bottom `entry_quantile`) e PERMANECE
        enquanto não sair da cauda mais larga (`exit_quantile`). Isso elimina o
        churn de fronteira — nomes oscilando na borda do quintil não geram trade.
        Sem estado anterior (prev_*) ou com exit==entry, recai no quintil puro.
        """
        # Percentil do score: 1.0 = maior (candidato a long), 0.0 = menor.
        pct = alpha_score.rank(pct=True, method="average")
        entry, exit_ = self.entry_quantile, self.exit_quantile

        # Entradas: caudas estreitas
        enter_long  = set(pct.index[pct >= 1.0 - entry])
        enter_short = set(pct.index[pct <= entry])

        if prev_long is not None and exit_ > entry:
            # Permanências: quem já estava na perna e não saiu da cauda larga
            stay_long  = set(pct.index[pct >= 1.0 - exit_]) & set(prev_long)
            stay_short = set(pct.index[pct <= exit_]) & set(prev_short)
            long_set  = enter_long | stay_long
            short_set = enter_short | stay_short
            # Desempate (raro): se um nome caísse nos dois, a entrada manda
            short_set -= long_set
        else:
            long_set, short_set = enter_long, enter_short

        long_tickers  = [t for t in alpha_score.index if t in long_set]
        short_tickers = [t for t in alpha_score.index if t in short_set]
        if not long_tickers or not short_tickers:
            logger.warning("Quintis long/short vazios — portfólio não construído.")
            return pd.Series(dtype=float)

        weights = pd.Series(0.0, index=full_index)
        weights[long_tickers]  = 1.0 / len(long_tickers)
        weights[short_tickers] = -self.long_short_ratio / len(short_tickers)
        return weights

    def _continuous_weights(
        self,
        alpha_score: pd.Series,
        full_index: pd.Index,
        cov: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        Peso contínuo sobre todo o cross-section.

        z-score : w_raw ∝ (α - ᾱ)/σ_α   (captura o IC inteiro)
        mv      : w_raw ∝ Σ⁻¹ · α_z      (Grinold; usa a estrutura de risco)

        Em ambos, w é centrado (dollar-neutral) antes do cap/renormalização.
        """
        std = alpha_score.std(ddof=0)
        if not np.isfinite(std) or std < 1e-12:
            logger.warning("Score sem dispersão cross-seccional — sem portfólio.")
            return pd.Series(dtype=float)
        a_z = (alpha_score - alpha_score.mean()) / std

        if cov is not None:
            common = [t for t in a_z.index if t in cov.index]
            if len(common) >= self.n_quantiles * 2:
                a_sub = a_z.loc[common]
                sigma = cov.loc[common, common].values
                try:
                    w_vec = np.linalg.solve(sigma, a_sub.values)
                    w_raw = pd.Series(w_vec, index=common)
                except np.linalg.LinAlgError:
                    logger.warning("Σ singular — caindo para z-score.")
                    w_raw = a_z
            else:
                w_raw = a_z
        else:
            w_raw = a_z

        # Dollar-neutral: centrar os pesos
        w_raw = w_raw - w_raw.mean()

        weights = pd.Series(0.0, index=full_index)
        weights[w_raw.index] = w_raw.values
        return weights

    def _estimate_cov(
        self,
        returns: pd.DataFrame,
        date: pd.Timestamp,
        tickers: list[str],
    ) -> Optional[pd.DataFrame]:
        """
        Covariância shrinkada (Ledoit-Wolf) na janela até `date`.

        LOOK-AHEAD SAFE: usa apenas retornos com índice ≤ date.
        Retorna None se não houver dados suficientes (o chamador cai para
        z-score). Ledoit-Wolf garante uma Σ bem-condicionada e invertível
        mesmo com nº de ativos próximo do nº de observações.
        """
        cols = [t for t in tickers if t in returns.columns]
        if len(cols) < self.n_quantiles * 2:
            return None
        window = returns.loc[returns.index <= date, cols].tail(self.cov_window)
        window = window.dropna(axis=1, thresh=self.cov_min_periods).dropna(axis=0)
        if window.shape[0] < self.cov_min_periods or window.shape[1] < self.n_quantiles * 2:
            return None
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(window.values)
            return pd.DataFrame(lw.covariance_, index=window.columns, columns=window.columns)
        except Exception as e:  # pragma: no cover - fallback defensivo
            logger.warning(f"Falha no Ledoit-Wolf ({e}); shrinkage manual p/ diagonal.")
            sample = window.cov()
            shrunk = 0.7 * sample + 0.3 * np.diag(np.diag(sample.values))
            return shrunk

    def build_weights_series(
        self,
        gds: pd.DataFrame,
        eligible_universe: Optional[dict[pd.Timestamp, list[str]]] = None,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Constrói pesos para todas as datas de rebalanceamento.

        Aplica a banda de no-trade entre datas consecutivas: nomes cujo
        peso-alvo mudou menos que `no_trade_band` mantêm o peso anterior,
        evitando turnover gerado por ruído do sinal.

        Parameters
        ----------
        returns : retornos diários (necessário p/ o método "mv"; ignorado
                  nos demais). Look-ahead safe — ver build_weights.

        Returns
        -------
        pd.DataFrame com pesos por data e ativo.
        index=rebalance_dates, cols=tickers.
        """
        all_weights = {}
        prev: Optional[pd.Series] = None
        prev_long: Optional[set] = None
        prev_short: Optional[set] = None

        for date in gds.index:
            gds_t = gds.loc[date]
            eligible = None
            if eligible_universe is not None:
                eligible = eligible_universe.get(date)

            target = self.build_weights(
                gds_t, eligible, returns=returns, date=date,
                prev_long=prev_long, prev_short=prev_short,
            )
            if target.empty:
                continue

            if prev is not None and self.no_trade_band > 0:
                target = self._apply_no_trade_band(target, prev)

            all_weights[date] = target
            prev = target
            # Estado p/ hysteresis: membership da perna (sinal do peso)
            prev_long  = set(target.index[target > 0])
            prev_short = set(target.index[target < 0])

        if not all_weights:
            raise RuntimeError("Nenhuma data gerou pesos válidos.")

        weights_df = pd.DataFrame(all_weights).T.fillna(0.0)
        weights_df.index.name = "Date"

        logger.info(
            f"Pesos construídos: {weights_df.shape[0]} datas × {weights_df.shape[1]} ativos | "
            f"método={self.method}, banda={self.no_trade_band:.3f} | "
            f"Pesos não-zero por data (média): "
            f"{(weights_df != 0).sum(axis=1).mean():.1f}"
        )
        return weights_df

    def _apply_no_trade_band(self, target: pd.Series, prev: pd.Series) -> pd.Series:
        """
        Mantém o peso anterior nos nomes cujo alvo mudou menos que a banda.

        new_i = prev_i        se |target_i − prev_i| < banda
              = target_i      caso contrário

        Depois renormaliza as pernas para preservar a neutralidade. Corta o
        turnover de ruído sem deixar a exposição bruta derivar.
        """
        union = target.index.union(prev.index)
        t = target.reindex(union).fillna(0.0)
        p = prev.reindex(union).fillna(0.0)

        hold = (t - p).abs() < self.no_trade_band
        new = t.copy()
        new[hold] = p[hold]

        new = new[new != 0.0]
        if new.empty:
            return target
        new = self._renormalize(new)
        return new

    # ──────────────────────────────────────────────────────────────
    # UTILITÁRIOS
    # ──────────────────────────────────────────────────────────────
    def _apply_weight_cap(self, weights: pd.Series) -> pd.Series:
        """
        Cap de peso máximo por ativo para evitar concentração.
        Excesso é redistribuído uniformemente nos outros ativos da mesma perna.

        Se o número de ativos é tão pequeno que peso igual já excede o cap
        (ex: 4 ativos com cap=15% → peso igual=25%), o cap não é aplicado
        pois forçaria uma restrição infactível. Isso é logado como aviso.
        """
        cap = self.max_weight

        def _cap_leg(leg_weights: pd.Series) -> pd.Series:
            """Aplica cap em uma perna (positiva ou negativa)."""
            n = len(leg_weights)
            if n == 0:
                return leg_weights
            # Se peso igual já excede o cap, não há como redistribuir
            # (restrição infactível) — mantém igual e loga aviso
            if 1.0 / n > cap:
                logger.debug(
                    f"Cap de {cap:.0%} infactível com {n} ativos "
                    f"(peso igual = {1/n:.2%}). Usando pesos iguais."
                )
                return leg_weights  # já normalizado
            # Redistribuição iterativa
            max_iters = 20
            for _ in range(max_iters):
                over = leg_weights > cap
                if not over.any():
                    break
                excess = (leg_weights[over] - cap).sum()
                leg_weights[over] = cap
                under = leg_weights[~over]
                if len(under) > 0:
                    leg_weights[~over] += excess / len(under)
            return leg_weights

        # Long leg
        long_mask = weights > 0
        if long_mask.sum() > 0:
            weights[long_mask] = _cap_leg(weights[long_mask].copy())

        # Short leg (trabalhar com positivos)
        short_mask = weights < 0
        if short_mask.sum() > 0:
            short_pos = (-weights[short_mask]).copy()
            short_pos = _cap_leg(short_pos)
            weights[short_mask] = -short_pos

        return weights

    def _renormalize(self, weights: pd.Series) -> pd.Series:
        """
        Renormaliza após o cap para garantir que:
          Σ w_long = +1.0
          Σ w_short = -long_short_ratio
        """
        long_mask  = weights > 0
        short_mask = weights < 0

        if long_mask.sum() > 0:
            long_sum = weights[long_mask].sum()
            if long_sum > 1e-9:
                weights[long_mask] /= long_sum  # soma a 1

        if short_mask.sum() > 0:
            short_sum = weights[short_mask].abs().sum()
            if short_sum > 1e-9:
                weights[short_mask] *= (self.long_short_ratio / short_sum)

        return weights

    def get_turnover(
        self,
        weights_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Calcula turnover entre rebalanceamentos consecutivos.

        Turnover(t) = Σ_i |w_i(t) - w_i(t-1)| / 2

        Turnover = 1.0 significa que todo o portfólio foi renovado.
        Usado para estimar custos de transação.
        """
        diff = weights_df.diff().abs().sum(axis=1) / 2.0
        diff.iloc[0] = weights_df.iloc[0].abs().sum() / 2.0  # primeiro rebal = comprar tudo
        return diff
