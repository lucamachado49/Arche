"""
collector.py
────────────
Coleta dados de preços da B3 usando fontes gratuitas.
Substitui o Bloomberg para o contexto da competição.

HIERARQUIA DE FONTES (por qualidade decrescente):
  1. B3 Cotações Históricas (arquivos .zip oficiais)
     - Cobre TODOS os ativos desde 1994, incluindo deslistados
     - Resolve parcialmente o survivorship bias
     - Não tem ajuste por dividendos/splits — requer tratamento manual
     - URL: https://bvmf.bmfbovespa.com.br/InstDados/SerHist/

  2. yfinance (Yahoo Finance)
     - Ajustado por dividendos e splits (auto_adjust=True)
     - Histórico desde ~2000 para a maioria dos ativos .SA
     - Survivorship bias: ativos deslistados retornam dados vazios
     - Instável: Yahoo aperta limites periodicamente (2024)

  3. brapi.dev
     - API brasileira, dados OHLCV + dividendos
     - Token gratuito no dashboard cobre >4.000 ativos
     - Mais estável que yfinance para ativos brasileiros
     - Histórico: varia por ativo (geralmente 5 anos)

ESTRATÉGIA DO PROJETO:
  - Usar B3 oficial para histórico bruto (2012–2024)
  - Usar yfinance para ajuste por proventos quando possível
  - Cruzar as duas fontes para detectar inconsistências
  - Documentar ativos com dados problemáticos em data/processed/data_quality.csv
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger

from .config import get, PROJECT_ROOT

# ─── Caminhos ──────────────────────────────────────────────────
RAW_B3    = PROJECT_ROOT / "data" / "raw" / "b3"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# ─── Universo IBrX-100 (B3, carteira teórica mai/2024) ────────
# Fonte: https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBRX100
# Cobertura: ~70-75% do volume total da B3, muito maior cross-section que Ibovespa
IBRX100_UNIVERSE = [
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

# Manter alias para compatibilidade retroativa
IBOVESPA_UNIVERSE = IBRX100_UNIVERSE


class B3DataCollector:
    """
    Coleta e integra dados de múltiplas fontes gratuitas.

    Uso típico
    ----------
    collector = B3DataCollector()

    # Método recomendado: baixa de todas as fontes e integra
    prices = collector.collect_all(
        tickers=IBOVESPA_UNIVERSE,
        start="2012-01-01",
        end="2024-12-31",
    )
    prices.to_parquet("data/raw/b3/prices_b3.parquet")
    """

    def __init__(self) -> None:
        self.cfg = get("universe")
        self.start = pd.Timestamp(self.cfg["start_date"])
        self.end   = pd.Timestamp(self.cfg["end_date"])

        # Rate limiting entre requests
        self._request_delay = 0.5  # segundos entre chamadas yfinance

    # ──────────────────────────────────────────────────────────────
    # 1. FONTE PRIMÁRIA — B3 Oficial (arquivos históricos)
    # ──────────────────────────────────────────────────────────────
    def download_b3_official(
        self,
        years: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Baixa arquivos históricos oficiais da B3 (BVMF).

        Os arquivos contêm TODOS os ativos negociados no dia,
        incluindo deslistados — resolve parcialmente survivorship bias.

        Formato: arquivo .txt de largura fixa por ano.
        Layout documentado em: https://bvmf.bmfbovespa.com.br/

        Parameters
        ----------
        years : lista de anos a baixar. Se None, usa 2012–2024.

        Returns
        -------
        pd.DataFrame com index=Date, cols=tickers, values=preço fechamento.
        ATENÇÃO: preços NÃO ajustados por proventos — ver adjust_with_yfinance().
        """
        if years is None:
            years = list(range(self.start.year, self.end.year + 1))

        all_data = []

        for year in years:
            logger.info(f"Baixando B3 oficial: ano {year}")
            url = f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"

            try:
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()

                # Ler ZIP em memória
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                txt_name = [n for n in zf.namelist() if n.endswith(".TXT")][0]

                with zf.open(txt_name) as f:
                    df_year = self._parse_b3_cotahist(f)

                all_data.append(df_year)
                logger.info(f"  {year}: {df_year.shape[1]} ativos, {len(df_year)} dias")
                time.sleep(1)  # respeitar servidor da B3

            except requests.HTTPError as e:
                logger.warning(f"  {year}: HTTP {e.response.status_code} — pulando")
            except Exception as e:
                logger.error(f"  {year}: Erro inesperado — {e}")

        if not all_data:
            raise RuntimeError("Nenhum dado baixado da B3. Verifique a conexão.")

        prices = pd.concat(all_data, axis=0).sort_index()

        # Remover duplicatas (podem ocorrer em feriados)
        prices = prices[~prices.index.duplicated(keep="last")]

        logger.info(
            f"B3 oficial consolidado: {prices.shape[1]} ativos × "
            f"{len(prices)} dias ({prices.index[0].date()} → {prices.index[-1].date()})"
        )
        return prices

    def _parse_b3_cotahist(self, file_obj) -> pd.DataFrame:
        """
        Parseia o arquivo COTAHIST da B3 (formato fixo .TXT).

        Layout COTAHIST (posições baseadas na documentação oficial B3):
          2       : tipo de registro (01=cotação diária)
          3–10    : data de pregão (AAAAMMDD)
          13–24   : código do papel
          49–52   : tipo de mercado (10=à vista)
          109–121 : preço de fechamento (11 dígitos, 2 decimais implícitos)
          153–170 : volume total negociado (16 dígitos, 2 decimais)
        """
        records = []

        for line in file_obj:
            if isinstance(line, bytes):
                line = line.decode("latin-1")

            # Apenas registros de cotação diária (tipo 01)
            if line[0:2] != "01":
                continue

            tipo_mercado = line[24:27].strip().zfill(3)
            # Apenas mercado à vista (010)
            if tipo_mercado != "010":
                continue

            try:
                data_dt  = pd.to_datetime(line[2:10], format="%Y%m%d", errors="coerce")
                if pd.isna(data_dt):
                    continue
                ticker   = line[12:24].strip()
                fechamento = float(line[108:121]) / 100.0  # 2 decimais implícitos
                volume   = float(line[152:170]) / 100.0

                # Filtrar tickers que parecem válidos (4-6 chars alfanuméricos)
                if not ticker or len(ticker) < 4 or not ticker[:4].isalpha():
                    continue

                records.append({
                    "Date":    data_dt,
                    "ticker":  ticker,
                    "close":   fechamento,
                    "volume":  volume,
                })
            except (ValueError, IndexError):
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Pivot: linhas = datas, colunas = tickers
        prices = df.pivot_table(
            index="Date",
            columns="ticker",
            values="close",
            aggfunc="last",  # se houver duplicata no dia, pega o último
        )
        prices.columns.name = None
        prices.index.name = "Date"

        return prices

    # ──────────────────────────────────────────────────────────────
    # 2. FONTE SECUNDÁRIA — yfinance (ajustado por proventos)
    # ──────────────────────────────────────────────────────────────
    def download_yfinance(
        self,
        tickers: list[str],
    ) -> pd.DataFrame:
        """
        Baixa preços ajustados via yfinance.

        Preços JÁ ajustados por dividendos e splits (auto_adjust=True).
        Ticker B3: adiciona sufixo .SA (VALE3 → VALE3.SA)

        LIMITAÇÕES CONHECIDAS:
        - Survivorship bias: deslistados retornam DataFrame vazio
        - Rate limiting: Yahoo pode bloquear >50 req/min
        - Confiabilidade: instável desde 2024

        Parameters
        ----------
        tickers    : lista de tickers sem sufixo (ex: ["VALE3", "PETR4"])

        Returns
        -------
        pd.DataFrame com preços ajustados (Close = Adj Close quando auto_adjust=True)
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance não instalado. Execute: pip install yfinance"
            )

        all_data = {}
        failed   = []

        # Download individual por ticker — mais robusto com yfinance >= 0.2.x
        # O batch download quebrou o MultiIndex em versões recentes
        logger.info(f"yfinance: baixando {len(tickers)} tickers individualmente...")

        for i, ticker in enumerate(tickers):
            yf_ticker = f"{ticker}.SA"

            if (i + 1) % 10 == 0:
                logger.info(f"  yfinance: {i+1}/{len(tickers)} ({ticker})")

            try:
                data = yf.download(
                    tickers=yf_ticker,
                    start=self.start.strftime("%Y-%m-%d"),
                    end=(self.end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                )
                # yfinance >= 0.2.38 retorna MultiIndex de colunas mesmo para um ticker
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                if data.empty:
                    failed.append(ticker)
                    continue

                # Coluna Close já é o preço ajustado quando auto_adjust=True
                if "Close" in data.columns:
                    series = data["Close"].dropna()
                elif "close" in data.columns:
                    series = data["close"].dropna()
                else:
                    failed.append(ticker)
                    continue

                if len(series) > 100:
                    all_data[ticker] = series
                else:
                    logger.warning(
                        f"  {ticker}: apenas {len(series)} obs — possível deslistagem"
                    )
                    failed.append(ticker)

            except Exception as e:
                logger.debug(f"  {ticker}: erro — {e}")
                failed.append(ticker)

            # Rate limiting: pausa entre tickers
            time.sleep(self._request_delay)

        if failed:
            logger.warning(
                f"yfinance: {len(failed)} tickers falharam ou têm dados insuficientes:\n"
                f"  {failed[:20]}{'...' if len(failed)>20 else ''}\n"
                f"  Possíveis causas: deslistagem, mudança de ticker, sem histórico."
            )

        if not all_data:
            raise RuntimeError("Nenhum dado baixado via yfinance.")

        prices = pd.DataFrame(all_data)
        prices.index.name = "Date"
        prices = prices.loc[self.start:self.end]

        logger.info(
            f"yfinance: {prices.shape[1]} ativos baixados com sucesso "
            f"({len(failed)} falharam)"
        )
        return prices

    # ──────────────────────────────────────────────────────────────
    # 3. FONTE AUXILIAR — brapi.dev
    # ──────────────────────────────────────────────────────────────
    def download_brapi(
        self,
        tickers: list[str],
        token: Optional[str] = None,
        range_: str = "5y",
    ) -> pd.DataFrame:
        """
        Baixa dados históricos via brapi.dev.

        Token gratuito disponível em: https://brapi.dev/dashboard
        Sem token: apenas PETR4, MGLU3, VALE3, ITUB4.

        Parameters
        ----------
        tickers : lista de tickers (sem sufixo)
        token   : token da brapi (ou via variável de ambiente BRAPI_TOKEN)
        range_  : período ("1y", "2y", "5y", "max")
        """
        import os
        token = token or os.getenv("BRAPI_TOKEN", "")

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        all_data = {}

        for ticker in tickers:
            try:
                url = f"https://brapi.dev/api/quote/{ticker}"
                resp = requests.get(
                    url,
                    params={"range": range_, "interval": "1d"},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()

                result = resp.json()["results"][0]
                hist   = result.get("historicalDataPrice", [])

                if not hist:
                    logger.warning(f"brapi: {ticker} sem dados históricos")
                    continue

                series = pd.Series(
                    {pd.Timestamp(d["date"], unit="s"): d["close"]
                     for d in hist if d.get("close") is not None},
                    name=ticker,
                )
                series.index.name = "Date"
                all_data[ticker] = series
                time.sleep(0.2)

            except Exception as e:
                logger.warning(f"brapi: {ticker} falhou — {e}")

        if not all_data:
            return pd.DataFrame()

        prices = pd.DataFrame(all_data)
        prices = prices.loc[self.start:self.end]
        logger.info(f"brapi: {prices.shape[1]} ativos coletados")
        return prices

    # ──────────────────────────────────────────────────────────────
    # 4. PIPELINE INTEGRADO — recomendado para o projeto
    # ──────────────────────────────────────────────────────────────
    def collect_all(
        self,
        tickers: Optional[list[str]] = None,
        use_b3_official: bool = True,
        use_yfinance: bool = True,
        brapi_token: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Pipeline completo de coleta integrando múltiplas fontes.

        Estratégia de integração:
          1. Baixar B3 oficial (histórico bruto, todos os ativos)
          2. Baixar yfinance (ajuste por proventos)
          3. Para cada ativo: usar yfinance como preço ajustado
             e B3 oficial apenas onde yfinance falha
          4. Documentar qualidade em data_quality.csv

        Parameters
        ----------
        tickers          : lista de tickers. Se None, usa IBOVESPA_UNIVERSE.
        use_b3_official  : baixar arquivos históricos da B3 (recomendado)
        use_yfinance     : baixar preços ajustados do Yahoo (recomendado)
        brapi_token      : token brapi.dev (opcional, melhora cobertura)

        Returns
        -------
        pd.DataFrame pronto para ser salvo em data/raw/b3/prices_b3.parquet
        """
        tickers = tickers or IBRX100_UNIVERSE
        # Remover duplicatas mantendo ordem
        tickers = list(dict.fromkeys(tickers))

        logger.info(
            f"Pipeline de coleta iniciado: {len(tickers)} ativos, "
            f"B3={use_b3_official}, yfinance={use_yfinance}"
        )

        frames = {}

        # ── Fonte 1: yfinance (ajustado) ──────────────────────
        if use_yfinance:
            try:
                yf_prices = self.download_yfinance(tickers)
                for col in yf_prices.columns:
                    frames[col] = yf_prices[col]
                logger.info(f"yfinance: {len(frames)} ativos coletados")
            except Exception as e:
                logger.error(f"yfinance falhou completamente: {e}")

        # ── Fonte 2: B3 oficial (complementar) ───────────────
        if use_b3_official:
            try:
                b3_prices = self.download_b3_official()
                # Usar B3 oficial para ativos não cobertos pelo yfinance
                yf_tickers = set(frames.keys())
                b3_added = 0
                for col in b3_prices.columns:
                    if col not in yf_tickers and col in tickers:
                        frames[col] = b3_prices[col]
                        b3_added += 1
                        logger.debug(f"  B3 oficial usado para: {col}")
                logger.info(f"B3 oficial: {b3_added} ativos adicionados")
            except Exception as e:
                logger.error(f"B3 oficial falhou: {e}")

        # ── Fonte 3: brapi (complementar) ─────────────────────
        # Tenta sempre — sem token, brapi fornece 4 tickers gratuitos úteis
        if True:  # noqa: SIM210 — intencional: tenta brapi independente do token
            missing = [t for t in tickers if t not in frames]
            if missing:
                try:
                    brapi_prices = self.download_brapi(
                        missing[:10],  # limitar para evitar rate limit
                        token=brapi_token,
                    )
                    for col in brapi_prices.columns:
                        if col not in frames:
                            frames[col] = brapi_prices[col]
                except Exception as e:
                    logger.warning(f"brapi falhou: {e}")

        if not frames:
            raise RuntimeError(
                "Nenhuma fonte de dados funcionou. "
                "Verifique a conexão com a internet."
            )

        # ── Consolidar ────────────────────────────────────────
        prices = pd.DataFrame(frames).sort_index()
        prices = prices.loc[self.start:self.end]

        # ── Relatório de qualidade ────────────────────────────
        self._generate_quality_report(prices, tickers)

        logger.info(
            f"\nColeta concluída:\n"
            f"  Ativos coletados   : {prices.shape[1]}/{len(tickers)}\n"
            f"  Ativos faltando    : "
            f"{[t for t in tickers if t not in prices.columns][:10]}\n"
            f"  Período            : {prices.index[0].date()} → {prices.index[-1].date()}\n"
            f"  Fill rate médio    : {prices.notna().mean().mean():.1%}"
        )

        return prices

    def _generate_quality_report(
        self,
        prices: pd.DataFrame,
        requested_tickers: list[str],
    ) -> None:
        """
        Gera relatório de qualidade dos dados coletados.
        Salvo em data/processed/data_quality.csv
        """
        PROCESSED.mkdir(parents=True, exist_ok=True)

        rows = []
        for ticker in requested_tickers:
            if ticker in prices.columns:
                series = prices[ticker]
                rows.append({
                    "ticker":        ticker,
                    "status":        "ok",
                    "n_obs":         series.notna().sum(),
                    "fill_rate":     series.notna().mean(),
                    "first_date":    series.dropna().index[0].date() if series.notna().any() else None,
                    "last_date":     series.dropna().index[-1].date() if series.notna().any() else None,
                    "pct_zero":      (series == 0).mean(),
                })
            else:
                rows.append({
                    "ticker":    ticker,
                    "status":    "missing",
                    "n_obs":     0,
                    "fill_rate": 0.0,
                })

        report = pd.DataFrame(rows).set_index("ticker")
        report.to_csv(PROCESSED / "data_quality.csv")
        logger.info(f"Relatório de qualidade salvo: data/processed/data_quality.csv")

        # Resumo
        n_ok      = (report["status"] == "ok").sum()
        n_missing = (report["status"] == "missing").sum()
        n_low     = (report.get("fill_rate", pd.Series([0])) < 0.5).sum()

        logger.warning(
            f"Qualidade dos dados:\n"
            f"  OK            : {n_ok}\n"
            f"  Não coletados : {n_missing}\n"
            f"  Fill rate <50%: {n_low}"
        )


# ──────────────────────────────────────────────────────────────
# SCRIPT DE COLETA — rodar diretamente
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Uso: python -m src.data.collector

    Variáveis de ambiente opcionais:
      BRAPI_TOKEN : token da brapi.dev (gratuito em brapi.dev/dashboard)
    """
    import os

    collector = B3DataCollector()

    prices = collector.collect_all(
        tickers=IBRX100_UNIVERSE,
        use_b3_official=True,
        use_yfinance=True,
        brapi_token=os.getenv("BRAPI_TOKEN"),
    )

    # Salvar
    RAW_B3.mkdir(parents=True, exist_ok=True)
    out = RAW_B3 / "prices_b3.parquet"
    prices.to_parquet(out)
    print(f"\n✓ Dados salvos em: {out}")
    print(f"  Shape: {prices.shape}")
    print(f"  Período: {prices.index[0].date()} → {prices.index[-1].date()}")
