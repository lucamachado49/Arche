"""
scripts/run_transfer_test.py
────────────────────────────
Pipeline completo do teste de hipótese cross-market (S&P 500 → Ibovespa).

Uso:
    python -m scripts.run_transfer_test

Fluxo:
  1. Carrega preços do Ibovespa (data/raw/b3/prices_b3.parquet)
  2. Baixa preços do S&P 500 (yfinance)
  3. Computa GDS INDEPENDENTE por mercado (geometrias isoladas)
  4. Roda a bateria de testes pooled (replicação + meta-análise + permutação)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.config import PROJECT_ROOT, get
from src.transfer.sp500 import download_sp500_prices
from src.transfer.cross_market import CrossMarketGDS
from src.transfer.pooled_test import CrossMarketHypothesisTest


def main() -> None:
    # ── 1. Preços do Ibovespa (já coletados pelo collector) ──
    ibov_path = PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet"
    if not ibov_path.exists():
        raise FileNotFoundError(
            f"{ibov_path} não encontrado. Rode antes: python -m src.data.collector"
        )
    prices_br = pd.read_parquet(ibov_path)
    logger.info(f"Ibovespa: {prices_br.shape[1]} ativos × {len(prices_br)} dias")

    # ── 2. Preços do S&P 500 ──
    limit = get("transfer", "sp500_max_tickers", 200)
    prices_us = download_sp500_prices(
        tickers=None,  # usa get_sp500_tickers(limit) internamente
    )

    # ── 3. GDS independente por mercado ──
    cm = CrossMarketGDS()
    panels = cm.compute_all(
        prices_by_market={"BR": prices_br, "US": prices_us},
        n_jobs=-1,
        use_cache=True,
    )

    # ── 4. Bateria de testes pooled ──
    test = CrossMarketHypothesisTest()
    results = test.run_full_battery(panels)

    # ── 5. Salvar resumo ──
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transfer_test_results.json"

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        return o

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_clean(results), f, indent=2, ensure_ascii=False)

    rep = results["replication"]
    print("\n" + "═" * 60)
    print("RESULTADO DO TESTE CROSS-MARKET (S&P → Ibovespa)")
    print("═" * 60)
    print(f"Replicação: descoberta={rep['discovery_market']} h*={rep['h_star']} "
          f"→ confirmação={rep['confirm_market']}")
    print(f"  IC confirmação: {rep['confirmation']['ic_mean']:.4f} "
          f"(t_NW={rep['confirmation']['t_nw']:.2f}, p_perm={rep['confirmation_perm_p']:.4f})")
    print(f"  PASSOU: {rep['passed']}")
    print(f"\nResumo completo salvo em: {out_path}")


if __name__ == "__main__":
    main()
