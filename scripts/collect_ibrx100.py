"""
scripts/collect_ibrx100.py
──────────────────────────
Script de coleta de dados para o universo IBrX-100.

Usa yf.Ticker.history() individualmente por ticker — a API mais robusta
do yfinance para ações brasileiras (evita problemas de batch/MultiIndex).

Uso:
    python scripts/collect_ibrx100.py

Saída:
    data/raw/b3/prices_ibrx100.parquet
    data/processed/data_quality_ibrx100.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Garantir UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from loguru import logger

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
START = "2012-01-01"
END   = "2024-12-31"
OUTPUT_PRICES  = PROJECT_ROOT / "data" / "raw" / "b3" / "prices_ibrx100.parquet"
OUTPUT_VOLUMES = PROJECT_ROOT / "data" / "raw" / "b3" / "volumes_ibrx100.parquet"
OUTPUT_QUALITY = PROJECT_ROOT / "data" / "processed" / "data_quality_ibrx100.csv"

# IBrX-100: carteira teórica (mai/2024)
IBRX100 = [
    # Commodities & Energia
    "VALE3", "PETR4", "PETR3", "PRIO3", "CSAN3", "UGPA3", "VBBR3",
    "RECV3", "RRRP3",
    # Financeiro — Bancos
    "ITUB4", "BBDC4", "BBAS3", "SANB11", "BPAC11", "BRSR6",
    # Financeiro — Seguros & Mercado
    "B3SA3", "IRBR3", "BBSE3", "CXSE3",
    # Financeiro — Outros
    "BRAP4", "BRKM5",
    # Consumo Básico
    "ABEV3", "BRFS3", "JBSS3", "MRFG3", "BEEF3", "SMTO3",
    "MDIA3", "NTCO3",
    # Consumo Discricionário & Varejo
    "MGLU3", "LREN3", "ARZZ3", "SOMA3", "VIVA3",
    "GMAT3", "PETZ3",
    # Industrial & Transporte
    "WEGE3", "EMBR3", "RAIL3", "CCRO3", "ECOR3",
    "AZUL4", "GOLL4", "RAIZ4",
    # Materiais & Papel
    "SUZB3", "KLBN11", "DXCO3", "CSNA3", "GGBR4", "USIM5",
    # Utilidades — Energia Elétrica
    "ELET3", "ELET6", "CPFE3", "CMIG4", "ENGI11", "EGIE3",
    "ENEV3", "EQTL3", "TAEE11", "CPLE6",
    # Utilidades — Saneamento & Gás
    "SBSP3", "SAPR11", "CSMG3",
    # Telecomunicações
    "VIVT3", "TIMS3",
    # Saúde
    "RDOR3", "HAPV3", "FLRY3", "DASA3", "GNDI3", "PARD3",
    # Tecnologia & Serviços
    "TOTS3", "LWSA3", "CASH3", "INTB3",
    # Imobiliário — Incorporadoras
    "CYRE3", "MRVE3", "EVEN3", "DIRR3", "PLPL3",
    # Imobiliário — FIIs (via ações ON)
    "HYPE3", "MULT3",
    # Logística & Infraestrutura
    "AURE3", "ANIM3",
    # Agro
    "SLCE3", "AGRO3", "TTEN3",
]


def _make_yf_session():
    """
    Cria sessão curl_cffi que impersona o Chrome.
    Necessário para contornar o rate-limiting do Yahoo Finance (2024+).
    """
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
        return session
    except Exception:
        return None


def collect_ticker(
    ticker: str,
    start: str,
    end: str,
    session=None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Baixa preços ajustados e volumes via yf.Ticker.history() com curl_cffi.

    curl_cffi impersona o Chrome e contorna o rate-limiting do Yahoo (2024+).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Execute: pip install yfinance")

    yf_ticker = f"{ticker}.SA"

    try:
        t = yf.Ticker(yf_ticker, session=session)
        hist = t.history(
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            raise_errors=False,
        )

        if hist is None or hist.empty or len(hist) < 100:
            return None, None

        # Normalizar índice: tz-aware → tz-naive
        if hasattr(hist.index, "tz") and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        hist.index = pd.to_datetime(hist.index)
        hist = hist.loc[start:end]

        if "Close" not in hist.columns:
            return None, None

        prices  = hist["Close"].rename(ticker)
        volumes = hist["Volume"].rename(ticker) if "Volume" in hist.columns else pd.Series(dtype=float, name=ticker)

        return prices, volumes

    except Exception as e:
        logger.debug(f"  {ticker}: {type(e).__name__}: {e}")
        return None, None



def run_collection() -> None:
    """Pipeline principal de coleta."""
    import os

    logger.info("=" * 60)
    logger.info("COLETA IBrX-100 via yfinance + curl_cffi (impersonate Chrome)")
    logger.info(f"Período: {START} → {END}")
    logger.info(f"Universo: {len(IBRX100)} ativos")
    logger.info("=" * 60)

    # Criar sessão curl_cffi uma vez e reutilizar
    session = _make_yf_session()
    if session is not None:
        logger.info("✓ curl_cffi session ativa (impersonate=chrome)")
    else:
        logger.warning("⚠ curl_cffi não disponível — usando requests padrão (pode falhar)")

    # Criar diretórios
    OUTPUT_PRICES.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_QUALITY.parent.mkdir(parents=True, exist_ok=True)

    prices_dict  = {}
    volumes_dict = {}
    quality      = []

    for i, ticker in enumerate(IBRX100):
        progress = f"[{i+1:3d}/{len(IBRX100)}]"

        prices, volumes = collect_ticker(ticker, START, END, session=session)

        if prices is not None:
            prices_dict[ticker]  = prices
            volumes_dict[ticker] = volumes

            fill_rate = prices.notna().mean()
            first     = prices.dropna().index[0].date()
            last      = prices.dropna().index[-1].date()
            n_obs     = prices.notna().sum()

            quality.append({
                "ticker":     ticker,
                "status":     "ok",
                "n_obs":      n_obs,
                "fill_rate":  round(fill_rate, 4),
                "first_date": first,
                "last_date":  last,
            })
            logger.info(f"{progress} ✓ {ticker}: {n_obs} obs ({first} → {last})")
        else:
            quality.append({
                "ticker":    ticker,
                "status":    "failed",
                "n_obs":     0,
                "fill_rate": 0.0,
            })
            logger.warning(f"{progress} ✗ {ticker}: sem dados")

        # Rate limiting
        time.sleep(0.4)

    # ── Consolidar ────────────────────────────────────────────
    if not prices_dict:
        logger.error("ERRO: nenhum ativo coletado.")
        sys.exit(1)

    prices_df  = pd.DataFrame(prices_dict).sort_index()
    volumes_df = pd.DataFrame(volumes_dict).sort_index()

    # Filtrar período
    prices_df  = prices_df.loc[START:END]
    volumes_df = volumes_df.loc[START:END]

    # Remover colunas com menos de 252 observações
    min_obs = 252
    valid_cols = prices_df.notna().sum() >= min_obs
    dropped = (~valid_cols).sum()
    if dropped > 0:
        logger.warning(f"Removendo {dropped} ativos com < {min_obs} obs: "
                       f"{list(prices_df.columns[~valid_cols])}")
    prices_df  = prices_df.loc[:, valid_cols]
    volumes_df = volumes_df.loc[:, prices_df.columns]

    # ── Salvar ────────────────────────────────────────────────
    prices_df.to_parquet(OUTPUT_PRICES)
    volumes_df.to_parquet(OUTPUT_VOLUMES)

    quality_df = pd.DataFrame(quality).set_index("ticker")
    quality_df.to_csv(OUTPUT_QUALITY)

    # ── Relatório final ───────────────────────────────────────
    n_ok     = (quality_df["status"] == "ok").sum()
    n_failed = (quality_df["status"] == "failed").sum()
    avg_fill = quality_df[quality_df["status"] == "ok"]["fill_rate"].mean()

    logger.info("")
    logger.info("=" * 60)
    logger.info("COLETA CONCLUÍDA")
    logger.info("=" * 60)
    logger.info(f"  Ativos coletados : {n_ok}/{len(IBRX100)}")
    logger.info(f"  Ativos falhados  : {n_failed}")
    logger.info(f"  Fill rate médio  : {avg_fill:.1%}")
    logger.info(f"  Shape final      : {prices_df.shape}")
    logger.info(f"  Período          : {prices_df.index[0].date()} → {prices_df.index[-1].date()}")
    logger.info(f"  Salvo em         : {OUTPUT_PRICES}")

    # Gate Sprint 0: fill rate > 70%
    gate_passed = avg_fill >= 0.70
    if gate_passed:
        logger.info("  Gate Sprint 0    : ✓ PASSOU (fill rate ≥ 70%)")
    else:
        logger.warning(f"  Gate Sprint 0    : ✗ FALHOU (fill rate = {avg_fill:.1%} < 70%)")

    print(f"\n✓ Dados salvos: {OUTPUT_PRICES}")
    print(f"  Shape: {prices_df.shape[0]} dias × {prices_df.shape[1]} ativos")
    print(f"  Período: {prices_df.index[0].date()} → {prices_df.index[-1].date()}")
    print(f"  Fill rate: {avg_fill:.1%}")
    if n_failed > 0:
        failed_list = quality_df[quality_df["status"] == "failed"].index.tolist()
        print(f"  Falharam ({n_failed}): {failed_list}")


if __name__ == "__main__":
    run_collection()
