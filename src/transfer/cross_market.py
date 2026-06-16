"""
src/transfer/cross_market.py
────────────────────────────
Computa o GDS de forma INDEPENDENTE para cada mercado.

INVARIANTE DE ISOLAMENTO (o que torna o transfer válido)
─────────────────────────────────────────────────────────
Cada mercado tem seu próprio grafo, sua própria curvatura, seu próprio
GDS — calculados SÓ com os ativos daquele mercado. Um ativo brasileiro
nunca entra no grafo americano e vice-versa. A geometria é market-specific;
só o GDS adimensional (z-score) é comparável entre mercados.

Reutiliza o SignalGenerator existente SEM modificá-lo: o pipeline de
geometria já é market-agnostic (recebe um DataFrame de retornos e calcula
tudo internamente). Basta alimentá-lo com preços de cada mercado.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.data.cleaner import DataCleaner
from src.data.config import get, PROJECT_ROOT
from src.signal.generator import SignalGenerator


class CrossMarketGDS:
    """
    Calcula (GDS, retornos) para múltiplos mercados, cada um na sua geometria.

    Uso típico
    ----------
    cm = CrossMarketGDS()
    panels = cm.compute_all({
        "BR": prices_ibov,
        "US": prices_sp500,
    })
    # panels["BR"] = (gds_br, returns_br)
    # panels["US"] = (gds_us, returns_us)
    """

    def __init__(self, cache_root: Optional[Path] = None) -> None:
        self.cleaner = DataCleaner()
        self.cache_root = cache_root or (PROJECT_ROOT / "data" / "cache" / "transfer")

    def compute_market(
        self,
        market: str,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        n_jobs: int = -1,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computa GDS e retornos para UM mercado.

        Parameters
        ----------
        market    : rótulo do mercado ("BR", "US", ...) — usado no cache.
        prices    : preços ajustados desse mercado (index=Date, cols=tickers).
        volumes   : volumes (opcional). Para o S&P, None ativa o proxy de
                    liquidez por consistência de preço — adequado a large caps.
        n_jobs    : paralelismo no cálculo da curvatura.
        use_cache : salva/carrega κ e GDS em disco por mercado.

        Returns
        -------
        (gds_ts, returns) : GDS por data/ativo e retornos diários do mercado.
        """
        logger.info(f"[{market}] Computando GDS na geometria própria do mercado...")

        cache_dir = (self.cache_root / market) if use_cache else None
        if cache_dir is not None:
            self._invalidate_stale_cache(market, cache_dir, prices)
        generator = SignalGenerator(cache_dir=cache_dir)

        # O SignalGenerator calcula retornos internamente para a curvatura,
        # mas precisamos dos retornos limpos também para o teste de IC.
        returns = self.cleaner.prices_to_log_returns(prices)

        _, gds_ts = generator.run(
            prices=prices,
            volumes=volumes,
            n_jobs=n_jobs,
            use_cache=use_cache,
        )

        logger.info(
            f"[{market}] GDS={gds_ts.shape} | retornos={returns.shape} | "
            f"{gds_ts.shape[1]} ativos no cross-section"
        )
        return gds_ts, returns

    def _invalidate_stale_cache(
        self,
        market: str,
        cache_dir: Path,
        prices: pd.DataFrame,
        min_coverage: float = 0.5,
    ) -> None:
        """
        Invalida cache de GDS incompatível com o universo de preços atual.

        Cenário real que motivou esta guarda: um run antigo com a lista de
        fallback do S&P (60 nomes) deixou cache que um run posterior com
        200 nomes reutilizou silenciosamente — o teste rodou com a
        interseção (~23 nomes), invalidando o braço US. Se as colunas do
        cache cobrem menos que `min_coverage` do universo de preços,
        apagamos o cache e forçamos recomputação.
        """
        gds_path = cache_dir / "gds_ts.parquet"
        kappa_path = cache_dir / "kappa_ts.parquet"
        if not gds_path.exists():
            return
        try:
            cached_cols = set(pd.read_parquet(gds_path).columns)
        except Exception as e:
            logger.warning(f"[{market}] cache ilegível ({e}) — removendo.")
            gds_path.unlink(missing_ok=True)
            kappa_path.unlink(missing_ok=True)
            return

        price_cols = set(prices.columns)
        coverage = len(cached_cols & price_cols) / max(1, len(price_cols))
        if coverage < min_coverage:
            logger.warning(
                f"[{market}] cache cobre só {coverage:.0%} do universo atual "
                f"({len(cached_cols)} colunas vs {len(price_cols)} preços) — "
                f"invalidando e recomputando."
            )
            gds_path.unlink(missing_ok=True)
            kappa_path.unlink(missing_ok=True)

    def compute_all(
        self,
        prices_by_market: dict[str, pd.DataFrame],
        volumes_by_market: Optional[dict[str, pd.DataFrame]] = None,
        n_jobs: int = -1,
        use_cache: bool = True,
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Computa (GDS, retornos) para todos os mercados.

        Returns
        -------
        dict {market: (gds_ts, returns)}
        """
        volumes_by_market = volumes_by_market or {}
        panels = {}
        for market, prices in prices_by_market.items():
            panels[market] = self.compute_market(
                market=market,
                prices=prices,
                volumes=volumes_by_market.get(market),
                n_jobs=n_jobs,
                use_cache=use_cache,
            )
        return panels
