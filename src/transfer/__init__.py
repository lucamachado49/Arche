"""
src/transfer/
─────────────
Transfer learning S&P 500 → Ibovespa para validação da hipótese do GDS.

PRINCÍPIO CENTRAL
─────────────────
NÃO transferimos a geometria (a curvatura do S&P é estruturalmente
diferente da brasileira — tech/Fed vs commodities/Selic). Transferimos
apenas a LEI ADIMENSIONAL:

    "Um colapso ≥ z·σ na curvatura nodal de Ricci precede reversão,
     em qualquer mercado guiado por correlações."

O GDS = Δκ/σ já é um z-score adimensional e market-relative. Por isso
um GDS de −2 significa o mesmo no S&P e no Ibovespa, mesmo que a
geometria subjacente difira. A curvatura é SEMPRE calculada dentro de
cada mercado (VALE3 nunca entra no grafo do S&P).

O S&P serve como amostra de REPLICAÇÃO independente da mesma lei,
dando o poder estatístico que ~86 ativos do Ibovespa não fornecem.

MÓDULOS
───────
  sp500.py        : aquisição do universo e preços do S&P 500
  cross_market.py : computa GDS independente por mercado (reusa SignalGenerator)
  pooled_test.py  : testes de hipótese pooled (replicação, meta-análise, permutação)
"""
from __future__ import annotations

from .cross_market import CrossMarketGDS
from .pooled_test import CrossMarketHypothesisTest

__all__ = ["CrossMarketGDS", "CrossMarketHypothesisTest"]
