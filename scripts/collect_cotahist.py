"""
scripts/collect_cotahist.py
───────────────────────────
Coleta preços históricos da B3 via COTAHIST oficial (2012-2024).

FONTE: https://bvmf.bmfbovespa.com.br/InstDados/SerHist/
  - Sem API key, sem rate-limit, sem autenticação
  - Cobre TODOS os ativos da B3 desde 1994 (inclui todo o IBrX-100)
  - 1 arquivo ZIP por ano (~30-80MB cada)

Confirmado funcionando: Status 200, parse correto do layout fixo.

Uso:
    python scripts/collect_cotahist.py

Saída:
    data/raw/b3/prices_ibrx100.parquet   (preços de fechamento)
    data/raw/b3/volumes_ibrx100.parquet  (volume financeiro R$)
    data/processed/data_quality_ibrx100.csv
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
from loguru import logger

# ── Configuração ──────────────────────────────────────────────
START_YEAR = 2012
END_YEAR   = 2024
BASE_URL   = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"

OUTPUT_DIR     = PROJECT_ROOT / "data" / "raw" / "b3"
OUTPUT_PRICES  = OUTPUT_DIR / "prices_ibrx100.parquet"
OUTPUT_VOLUMES = OUTPUT_DIR / "volumes_ibrx100.parquet"
OUTPUT_QUALITY = PROJECT_ROOT / "data" / "processed" / "data_quality_ibrx100.csv"

# IBrX-100 — carteira teórica mai/2024
IBRX100 = [
    # Commodities & Energia
    "VALE3", "PETR4", "PETR3", "PRIO3", "CSAN3", "UGPA3", "VBBR3",
    "RECV3", "RRRP3",
    # Financeiro — Bancos
    "ITUB4", "BBDC4", "BBAS3", "SANB11", "BPAC11", "BRSR6",
    # Financeiro — Seguros & Mercado de Capitais
    "B3SA3", "IRBR3", "BBSE3", "CXSE3",
    # Financeiro — Outros
    "BRAP4", "BRKM5",
    # Consumo Básico
    "ABEV3", "BRFS3", "JBSS3", "MRFG3", "BEEF3", "SMTO3", "MDIA3", "NTCO3",
    # Consumo Discricionário & Varejo
    "MGLU3", "LREN3", "ARZZ3", "SOMA3", "VIVA3", "GMAT3", "PETZ3",
    # Industrial & Transporte
    "WEGE3", "EMBR3", "RAIL3", "CCRO3", "ECOR3", "AZUL4", "GOLL4", "RAIZ4",
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
    # Imobiliário
    "CYRE3", "MRVE3", "EVEN3", "DIRR3", "PLPL3", "HYPE3", "MULT3",
    # Logística & Infraestrutura
    "AURE3", "ANIM3",
    # Agro
    "SLCE3", "AGRO3", "TTEN3",
]

IBRX100_SET = set(IBRX100)


def parse_cotahist_year(resp_content: bytes) -> pd.DataFrame:
    """
    Parseia um arquivo COTAHIST anual e retorna DataFrame com
    Date, ticker, close, volume — filtrado para IBrX-100.

    Layout COTAHIST (posições base-1, convertidas para base-0):
      [0:2]    tipo de registro (01 = cotação diária)
      [2:10]   data pregão (AAAAMMDD)
      [12:24]  código do papel
      [24:27]  tipo de mercado (010 = à vista)
      [108:121] preço de fechamento (11 dígitos, 2 decimais implícitos)
      [152:170] volume financeiro (16 dígitos, 2 decimais implícitos)
    """
    records = []
    zf = zipfile.ZipFile(io.BytesIO(resp_content))
    txt_name = next(n for n in zf.namelist() if n.endswith(".TXT"))

    with zf.open(txt_name) as f:
        for line in f:
            line = line.decode("latin-1")
            if line[0:2] != "01":
                continue
            if line[24:27].strip().zfill(3) != "010":
                continue

            ticker = line[12:24].strip()
            if ticker not in IBRX100_SET:
                continue

            try:
                data_dt    = pd.to_datetime(line[2:10], format="%Y%m%d", errors="coerce")
                if pd.isna(data_dt):
                    continue
                fechamento = float(line[108:121]) / 100.0
                volume     = float(line[152:170]) / 100.0
                records.append({
                    "Date":   data_dt,
                    "ticker": ticker,
                    "close":  fechamento,
                    "volume": volume,
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(records)


def run_collection() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("COTAHIST B3 — coleta histórica IBrX-100")
    logger.info(f"Anos: {START_YEAR} → {END_YEAR}")
    logger.info(f"Tickers IBrX-100: {len(IBRX100)}")
    logger.info("=" * 60)

    all_dfs = []
    years_ok, years_fail = [], []

    for year in range(START_YEAR, END_YEAR + 1):
        url = BASE_URL.format(year=year)
        logger.info(f"[{year}] Baixando {url.split('/')[-1]}...")

        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()

            df_year = parse_cotahist_year(resp.content)

            if df_year.empty:
                logger.warning(f"[{year}] Nenhum dado para os tickers IBrX-100")
                years_fail.append(year)
                continue

            n_dias    = df_year["Date"].nunique()
            n_tickers = df_year["ticker"].nunique()
            logger.info(f"[{year}] ✓ {n_dias} pregões × {n_tickers} ativos")

            all_dfs.append(df_year)
            years_ok.append(year)

        except requests.HTTPError as e:
            logger.warning(f"[{year}] HTTP {e.response.status_code} — pulando")
            years_fail.append(year)
        except Exception as e:
            logger.error(f"[{year}] Erro: {e}")
            years_fail.append(year)

        time.sleep(1.0)  # respeitar servidor B3

    if not all_dfs:
        logger.error("Nenhum dado coletado. Verifique a conexão.")
        sys.exit(1)

    # ── Consolidar ────────────────────────────────────────────
    logger.info("Consolidando dados...")
    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values(["Date", "ticker"])
    full = full.drop_duplicates(subset=["Date", "ticker"], keep="last")

    prices = full.pivot_table(
        index="Date", columns="ticker", values="close", aggfunc="last"
    )
    prices.columns.name = None
    prices.index.name = "Date"

    volumes = full.pivot_table(
        index="Date", columns="ticker", values="volume", aggfunc="last"
    )
    volumes.columns.name = None
    volumes.index.name = "Date"

    # ── Qualidade ─────────────────────────────────────────────
    quality_rows = []
    for ticker in IBRX100:
        if ticker in prices.columns:
            s = prices[ticker].dropna()
            quality_rows.append({
                "ticker":     ticker,
                "status":     "ok",
                "n_obs":      len(s),
                "fill_rate":  round(prices[ticker].notna().mean(), 4),
                "first_date": s.index[0].date() if len(s) else None,
                "last_date":  s.index[-1].date() if len(s) else None,
            })
        else:
            quality_rows.append({
                "ticker":    ticker,
                "status":    "missing",
                "n_obs":     0,
                "fill_rate": 0.0,
            })

    quality_df = pd.DataFrame(quality_rows).set_index("ticker")

    # Remover tickers com < 252 observações
    valid = quality_df["n_obs"] >= 252
    prices  = prices[[c for c in prices.columns if valid.get(c, False)]]
    volumes = volumes[[c for c in volumes.columns if c in prices.columns]]

    # ── Salvar ────────────────────────────────────────────────
    prices.to_parquet(OUTPUT_PRICES)
    volumes.to_parquet(OUTPUT_VOLUMES)
    quality_df.to_csv(OUTPUT_QUALITY)

    # ── Relatório ─────────────────────────────────────────────
    n_ok  = (quality_df["status"] == "ok").sum()
    n_mis = (quality_df["status"] == "missing").sum()
    avg_fill = quality_df[quality_df["status"] == "ok"]["fill_rate"].mean()

    logger.info("")
    logger.info("=" * 60)
    logger.info("COLETA CONCLUÍDA")
    logger.info("=" * 60)
    logger.info(f"  Anos coletados   : {years_ok}")
    if years_fail:
        logger.warning(f"  Anos com falha   : {years_fail}")
    logger.info(f"  Ativos ok        : {n_ok}/{len(IBRX100)}")
    logger.info(f"  Ativos ausentes  : {n_mis}")
    logger.info(f"  Fill rate médio  : {avg_fill:.1%}")
    logger.info(f"  Shape final      : {prices.shape}")
    logger.info(f"  Período          : {prices.index[0].date()} → {prices.index[-1].date()}")
    logger.info(f"  Arquivo prices   : {OUTPUT_PRICES}")
    logger.info(f"  Arquivo volumes  : {OUTPUT_VOLUMES}")
    logger.info(f"  Qualidade        : {OUTPUT_QUALITY}")

    gate = avg_fill >= 0.70
    logger.info(f"  Gate Sprint 0    : {'✓ PASSOU' if gate else '✗ FALHOU'} (fill ≥ 70%)")

    print(f"\n✓ Concluído!")
    print(f"  {prices.shape[0]} pregões × {prices.shape[1]} ativos")
    print(f"  {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Fill rate médio: {avg_fill:.1%}")
    if n_mis > 0:
        missing = quality_df[quality_df["status"] == "missing"].index.tolist()
        print(f"  Ausentes ({n_mis}): {missing}")


if __name__ == "__main__":
    run_collection()
