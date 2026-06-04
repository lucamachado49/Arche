"""
src/transfer/sp500.py
─────────────────────
Aquisição do universo e preços do S&P 500 para a amostra de replicação.

Tickers US não usam sufixo (.SA é só B3). yfinance é estável e limpo
para ações americanas — sem os problemas de cobertura do mercado BR.

NOTA SOBRE SURVIVORSHIP BIAS
─────────────────────────────
A lista da Wikipédia é a composição ATUAL do índice. Para um estudo de
RETORNO absoluto isso enviesaria para cima. Aqui testamos uma lei de
REVERSÃO CROSS-SECCIONAL (rank IC) — muito menos sensível a survivorship,
pois o sinal é relativo entre os ativos vivos em cada data. Ainda assim,
para rigor máximo, troque get_sp500_tickers() por uma fonte point-in-time
de constituintes (ex.: dados históricos do índice).
"""
from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from src.data.config import get

# Fallback caso a leitura da Wikipédia falhe (sem lxml/conexão).
# ~60 large caps líquidas e estáveis — suficiente para um cross-section
# muito maior que o Ibovespa.
_FALLBACK_SP500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "BAC", "XOM", "CVX", "ABBV", "PFE",
    "KO", "PEP", "COST", "WMT", "MRK", "TMO", "CSCO", "ACN", "MCD", "ABT",
    "DHR", "ADBE", "WFC", "TXN", "NEE", "LIN", "PM", "ORCL", "CRM", "AMD",
    "HON", "UNP", "QCOM", "INTC", "IBM", "GE", "CAT", "GS", "MS", "RTX",
    "LOW", "SBUX", "BA", "BLK", "AMAT", "DE", "AXP", "BKNG", "GILD", "MDT",
]


def get_sp500_tickers(limit: Optional[int] = None) -> list[str]:
    """
    Retorna os tickers do S&P 500.

    Tenta ler a composição atual da Wikipédia (via pandas.read_html).
    Se falhar, usa a lista de fallback de large caps.

    Parameters
    ----------
    limit : se informado, retorna apenas os primeiros N tickers.
            Use para limitar o custo computacional da curvatura
            (200 já triplica o cross-section do Ibovespa).

    Returns
    -------
    Lista de tickers no formato do yfinance (ex.: "BRK-B", não "BRK.B").
    """
    url = get("transfer", "sp500_tickers_url",
              "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    try:
        # A Wikipédia bloqueia o User-Agent padrão do urllib (HTTP 403).
        # Buscamos via requests com UA de navegador e passamos o HTML ao pandas.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        # yfinance usa "-" no lugar de "." (BRK.B → BRK-B)
        tickers = (
            df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
        )
        logger.info(f"S&P 500: {len(tickers)} tickers lidos da Wikipédia.")
    except Exception as e:
        logger.warning(
            f"Falha ao ler tickers do S&P 500 ({e}). "
            f"Usando lista de fallback ({len(_FALLBACK_SP500)} large caps)."
        )
        tickers = _FALLBACK_SP500

    if limit is not None:
        tickers = tickers[:limit]
    return tickers


def download_sp500_prices(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """
    Baixa preços ajustados (auto_adjust=True) do S&P 500 via yfinance.

    Faz download em lotes (chunks) para robustez. Trata o MultiIndex de
    colunas que o yfinance >= 0.2.38 retorna.

    Parameters
    ----------
    tickers    : lista de tickers. Se None, usa get_sp500_tickers(limit).
    start, end : datas (YYYY-MM-DD). Se None, usa o período do settings.yaml.
    chunk_size : nº de tickers por requisição.

    Returns
    -------
    pd.DataFrame de preços ajustados (index=Date, cols=tickers).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance não instalado. Execute: pip install yfinance")

    cfg = get("universe")
    start = start or cfg["start_date"]
    end = end or cfg["end_date"]

    if tickers is None:
        limit = get("transfer", "sp500_max_tickers", 200)
        tickers = get_sp500_tickers(limit=limit)

    # yfinance: end é exclusivo — somar 1 dia para incluir a data final
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(
        f"S&P 500: baixando {len(tickers)} tickers de {start} a {end} "
        f"em lotes de {chunk_size}..."
    )

    frames = []
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i + chunk_size]
        try:
            data = yf.download(
                tickers=batch,
                start=start,
                end=end_inclusive,
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
        except Exception as e:
            logger.warning(f"  lote {i//chunk_size + 1}: erro — {e}")
            continue

        if data is None or data.empty:
            continue

        # Extrair os fechamentos (Close já é ajustado com auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex (campo, ticker) → fatiar "Close"
            if "Close" in data.columns.get_level_values(0):
                close = data["Close"].copy()
            else:
                continue
        else:
            # Lote de 1 ticker → colunas simples
            if "Close" in data.columns:
                close = data[["Close"]].copy()
                close.columns = [batch[0]]
            else:
                continue

        frames.append(close)
        logger.info(f"  lote {i//chunk_size + 1}: {close.shape[1]} ativos OK")

    if not frames:
        raise RuntimeError("Nenhum dado do S&P 500 baixado. Verifique a conexão.")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index.name = "Date"
    prices = prices.dropna(how="all", axis=1).dropna(how="all", axis=0)
    prices = prices.loc[str(start):str(end)]

    logger.info(
        f"S&P 500 consolidado: {prices.shape[1]} ativos × {len(prices)} dias "
        f"({prices.index[0].date()} → {prices.index[-1].date()})"
    )
    return prices
