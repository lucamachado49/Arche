"""
loader.py
─────────
Responsabilidade única: ler arquivos brutos e devolver DataFrames
validados, sem nenhuma transformação de negócio.

Suporta três fontes:
  1. Bloomberg export (CSV com múltiplos tickers)
  2. NEFIN (fatores de risco brasileiros — .xlsx ou .csv)
  3. Banco Central do Brasil (BCB via API pública)

INVARIANTE DE SEGURANÇA
───────────────────────
Este módulo NUNCA usa datas futuras.
Toda operação de leitura é sobre arquivos estáticos já exportados.
O teste tests/test_lookahead.py verifica isso formalmente.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger

from .config import get, PROJECT_ROOT


# ─── Caminhos base ─────────────────────────────────────────────
RAW_B3     = PROJECT_ROOT / "data" / "raw" / "b3"
RAW_SP500  = PROJECT_ROOT / "data" / "raw" / "sp500"
RAW_NEFIN  = PROJECT_ROOT / "data" / "raw" / "nefin"
RAW_BCB    = PROJECT_ROOT / "data" / "raw" / "bcb"
PROCESSED  = PROJECT_ROOT / "data" / "processed"


class DataLoader:
    """
    Carrega dados brutos e retorna DataFrames limpos para uso
    pelos módulos subsequentes.

    Uso típico
    ----------
    loader = DataLoader()
    prices = loader.load_b3_prices()
    factors = loader.load_nefin_factors()
    """

    def __init__(self) -> None:
        self.cfg_universe  = get("universe")
        self.cfg_paths     = get("paths")
        self.start_date    = pd.Timestamp(self.cfg_universe["start_date"])
        self.end_date      = pd.Timestamp(self.cfg_universe["end_date"])

    # ─────────────────────────────────────────────────────────────
    # 1. DADOS B3 — Preços de fechamento ajustados
    # ─────────────────────────────────────────────────────────────
    def load_b3_prices(
        self,
        source_file: Optional[str] = None,
        price_col: str = "PX_LAST",
    ) -> pd.DataFrame:
        """
        Carrega preços de fechamento ajustados da B3.

        O arquivo Bloomberg exportado deve ser um CSV com:
          - Coluna 'Date' (index)
          - Uma coluna por ticker (ex: "VALE3 BZ Equity")
          - Valores ausentes representados por '#N/A N/A' ou NaN

        Parameters
        ----------
        source_file : nome do arquivo dentro de data/raw/b3/.
                      Se None, tenta encontrar automaticamente.
        price_col   : nome da coluna de preço no export do Bloomberg.

        Returns
        -------
        pd.DataFrame com index DatetimeIndex e colunas = tickers limpos.
        Valores: preço de fechamento ajustado (float64).
        """
        path = self._resolve_path(RAW_B3, source_file, extensions=[".csv", ".parquet"])

        logger.info(f"Carregando preços B3 de: {path}")

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = self._read_bloomberg_csv(path)

        df = self._standardize_index(df)
        df = self._filter_date_range(df)
        df = self._clean_bloomberg_na(df)

        # Remover colunas totalmente vazias
        n_before = df.shape[1]
        df = df.dropna(axis=1, how="all")
        if df.shape[1] < n_before:
            logger.warning(
                f"Removidas {n_before - df.shape[1]} colunas completamente vazias"
            )

        self._validate_prices(df)
        logger.info(
            f"B3 carregado: {df.shape[1]} ativos | "
            f"{df.index[0].date()} → {df.index[-1].date()} "
            f"({len(df)} dias)"
        )
        return df

    def _read_bloomberg_csv(self, path: Path) -> pd.DataFrame:
        """
        Lê CSV no formato padrão de export do Bloomberg Terminal.
        Trata dois formatos comuns:
          - Formato 1: data na primeira coluna, tickers no header
          - Formato 2: duas linhas de header (nome + field)
        """
        # Tenta detectar o formato lendo as primeiras linhas
        with open(path, "r", encoding="utf-8-sig") as f:
            first_lines = [f.readline() for _ in range(5)]

        # Bloomberg às vezes exporta com header duplo
        if any("Dates" in line or "Date" in line for line in first_lines[:2]):
            df = pd.read_csv(
                path,
                index_col=0,
                na_values=["#N/A N/A", "#N/A", "N/A", "", " "],
                dtype=float,
                thousands=",",
            )
        else:
            # Formato com duas linhas de header: pula a primeira
            df = pd.read_csv(
                path,
                index_col=0,
                header=[0, 1],
                na_values=["#N/A N/A", "#N/A", "N/A", "", " "],
            )
            # Achata o MultiIndex de colunas
            df.columns = [f"{a}_{b}" if b else a for a, b in df.columns]
            df = df.astype(float)

        # Limpar nomes de colunas: "VALE3 BZ Equity" → "VALE3"
        df.columns = self._clean_ticker_names(df.columns.tolist())
        return df

    def _clean_ticker_names(self, tickers: list[str]) -> list[str]:
        """
        Remove sufixos Bloomberg: "VALE3 BZ Equity" → "VALE3"
        """
        suffixes_to_remove = [
            " BZ Equity", " US Equity", " BS Equity",
            " BZ", " US", " Equity", "_PX_LAST", "_LAST",
        ]
        cleaned = []
        for t in tickers:
            t_clean = str(t).strip()
            for suffix in suffixes_to_remove:
                if t_clean.upper().endswith(suffix.upper()):
                    t_clean = t_clean[:-len(suffix)].strip()
            cleaned.append(t_clean)
        return cleaned

    # ─────────────────────────────────────────────────────────────
    # 2. DADOS S&P 500 — Para Transfer Learning (Fase 2)
    # ─────────────────────────────────────────────────────────────
    def load_sp500_prices(
        self,
        source_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Carrega preços S&P 500. Mesmo formato que B3.
        """
        path = self._resolve_path(RAW_SP500, source_file, extensions=[".csv", ".parquet"])
        logger.info(f"Carregando preços S&P 500 de: {path}")

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = self._read_bloomberg_csv(path)

        df = self._standardize_index(df)
        df = self._filter_date_range(df)
        df = self._clean_bloomberg_na(df)
        df = df.dropna(axis=1, how="all")

        self._validate_prices(df)
        logger.info(f"S&P 500 carregado: {df.shape[1]} ativos | {len(df)} dias")
        return df

    # ─────────────────────────────────────────────────────────────
    # 3. FATORES NEFIN — Fama-French calibrado para Brasil
    # ─────────────────────────────────────────────────────────────
    def load_nefin_factors(
        self,
        source_file: Optional[str] = None,
        download_if_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Carrega fatores de risco do NEFIN (FEA/USP).

        Fatores incluídos:
          Rm_minus_Rf : excesso de retorno do mercado
          SMB         : Small Minus Big (tamanho)
          HML         : High Minus Low (valor)
          WML         : Winners Minus Losers (momentum)
          Rf          : taxa livre de risco (CDI)

        O NEFIN disponibiliza em:
        http://nefin.com.br/Risk_Factors/

        Returns
        -------
        pd.DataFrame com index DatetimeIndex e colunas = fatores.
        Valores em decimal (0.01 = 1%).
        """
        path = self._resolve_path(
            RAW_NEFIN, source_file,
            extensions=[".csv", ".xlsx", ".parquet"],
            raise_if_missing=not download_if_missing,
        )

        if path is None and download_if_missing:
            logger.info("Fatores NEFIN não encontrados localmente. Baixando...")
            df = self._download_nefin_factors()
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".xlsx":
            df = self._read_nefin_xlsx(path)
        else:
            df = pd.read_csv(path, index_col=0, parse_dates=True)

        df = self._standardize_index(df)
        df = self._filter_date_range(df)

        # Garantir que fatores estão em decimal, não em %
        if (df.abs() > 1).any().any():
            logger.warning(
                "Fatores NEFIN parecem estar em %. Dividindo por 100."
            )
            df = df / 100.0

        self._validate_factors(df)
        logger.info(f"NEFIN carregado: {list(df.columns)} | {len(df)} dias")
        return df

    def _read_nefin_xlsx(self, path: Path) -> pd.DataFrame:
        """Lê o formato Excel padrão do NEFIN."""
        dfs = {}
        factor_map = {
            "Market_Factor": "Rm_minus_Rf",
            "SMB_Factor": "SMB",
            "HML_Factor": "HML",
            "WML_Factor": "WML",
            "Risk_Free": "Rf",
        }
        for sheet_raw, col_name in factor_map.items():
            try:
                sheet = pd.read_excel(
                    path,
                    sheet_name=sheet_raw,
                    index_col=0,
                    parse_dates=True,
                )
                # NEFIN: coluna de retorno geralmente é a segunda
                ret_col = sheet.columns[0]
                dfs[col_name] = sheet[ret_col]
            except Exception as e:
                logger.warning(f"Aba '{sheet_raw}' não encontrada: {e}")

        df = pd.DataFrame(dfs)
        df.index.name = "Date"
        return df

    def _download_nefin_factors(self) -> pd.DataFrame:
        """
        Baixa os fatores diretamente do site do NEFIN.
        Requer conexão com internet.
        """
        urls = {
            "Rm_minus_Rf": "http://nefin.com.br/Risk_Factors/Market_Factor.xls",
            "SMB":          "http://nefin.com.br/Risk_Factors/SMB_Factor.xls",
            "HML":          "http://nefin.com.br/Risk_Factors/HML_Factor.xls",
            "WML":          "http://nefin.com.br/Risk_Factors/WML_Factor.xls",
            "Rf":           "http://nefin.com.br/Risk_Factors/Risk_Free.xls",
        }
        dfs = {}
        for factor, url in urls.items():
            try:
                logger.info(f"Baixando fator NEFIN: {factor}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                xls = pd.read_excel(
                    io.BytesIO(resp.content),
                    index_col=0,
                    parse_dates=True,
                )
                ret_col = xls.columns[0]
                dfs[factor] = xls[ret_col]
                # Salvar localmente para cache
                out = RAW_NEFIN / f"{factor}.csv"
                xls[[ret_col]].rename(columns={ret_col: factor}).to_csv(out)
                logger.info(f"Salvo em: {out}")
            except Exception as e:
                logger.error(f"Falha ao baixar {factor}: {e}")

        return pd.DataFrame(dfs)

    # ─────────────────────────────────────────────────────────────
    # 4. DADOS BCB — Taxa Selic e IPCA (para módulo macro)
    # ─────────────────────────────────────────────────────────────
    def load_selic(self, download_if_missing: bool = True) -> pd.Series:
        """
        Carrega taxa Selic diária do BCB.
        Usa API pública: https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados

        Returns
        -------
        pd.Series com index DatetimeIndex, valores em decimal ao dia.
        """
        path = RAW_BCB / "selic_diaria.parquet"

        if path.exists():
            selic = pd.read_parquet(path).squeeze()
            logger.info(f"Selic carregada localmente: {len(selic)} dias")
        elif download_if_missing:
            logger.info("Baixando Selic do BCB...")
            selic = self._download_bcb_series(series_code=11, name="selic")
            selic.to_frame().to_parquet(path)
        else:
            raise FileNotFoundError(f"Dados Selic não encontrados em {path}")

        selic = selic / 100.0  # % → decimal
        return selic.loc[self.start_date:self.end_date]

    def _download_bcb_series(self, series_code: int, name: str) -> pd.Series:
        """Baixa série temporal do SGS/BCB via API REST."""
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados"
            f"?formato=json"
            f"&dataInicial={self.start_date.strftime('%d/%m/%Y')}"
            f"&dataFinal={self.end_date.strftime('%d/%m/%Y')}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        s = pd.Series(
            {pd.Timestamp(d["data"], dayfirst=True): float(d["valor"])
             for d in data},
            name=name,
        )
        s.index.name = "Date"
        return s

    # ─────────────────────────────────────────────────────────────
    # UTILITÁRIOS PRIVADOS
    # ─────────────────────────────────────────────────────────────
    def _resolve_path(
        self,
        directory: Path,
        filename: Optional[str],
        extensions: list[str],
        raise_if_missing: bool = True,
    ) -> Optional[Path]:
        """
        Resolve o caminho do arquivo de dados.
        Se filename for None, procura automaticamente no diretório.
        """
        if filename:
            path = directory / filename
            if not path.exists():
                if raise_if_missing:
                    raise FileNotFoundError(f"Arquivo não encontrado: {path}")
                return None
            return path

        # Busca automática por extensão
        for ext in extensions:
            candidates = list(directory.glob(f"*{ext}"))
            if candidates:
                if len(candidates) > 1:
                    logger.warning(
                        f"Múltiplos arquivos {ext} em {directory}. "
                        f"Usando o mais recente: {candidates[-1].name}"
                    )
                return candidates[-1]

        if raise_if_missing:
            raise FileNotFoundError(
                f"Nenhum arquivo {extensions} encontrado em {directory}.\n"
                f"Exporte os dados do Bloomberg e salve em: {directory}"
            )
        return None

    def _standardize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garante que o index é DatetimeIndex com frequência de negócios."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, dayfirst=True)
        df.index.name = "Date"
        df = df.sort_index()
        # Remove duplicatas de datas
        n_dup = df.index.duplicated().sum()
        if n_dup > 0:
            logger.warning(f"Removendo {n_dup} datas duplicadas no index")
            df = df[~df.index.duplicated(keep="last")]
        return df

    def _filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra pelo range de datas do settings.yaml."""
        return df.loc[self.start_date:self.end_date]

    def _clean_bloomberg_na(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte strings de NA do Bloomberg para np.nan."""
        na_strings = ["#N/A N/A", "#N/A", "N/A", "N.A.", "#VALUE!"]
        df = df.replace(na_strings, np.nan)
        return df.astype(float)

    def _validate_prices(self, df: pd.DataFrame) -> None:
        """
        Valida sanidade dos preços carregados.
        Lança ValueError se algo crítico estiver errado.
        """
        if df.empty:
            raise ValueError("DataFrame de preços está vazio após filtragem.")

        if df.shape[1] < 10:
            raise ValueError(
                f"Apenas {df.shape[1]} ativos carregados. "
                f"Esperado mínimo de 10. Verifique o arquivo de entrada."
            )

        # Preços negativos são impossíveis
        neg_mask = (df < 0) & df.notna()
        if neg_mask.any().any():
            neg_tickers = df.columns[neg_mask.any()].tolist()
            raise ValueError(
                f"Preços negativos encontrados em: {neg_tickers}. "
                f"Verifique o ajuste por dividendos."
            )

        # Taxa de preenchimento mínima
        fill_rate = df.notna().mean().mean()
        if fill_rate < 0.5:
            logger.warning(
                f"Taxa de preenchimento baixa: {fill_rate:.1%}. "
                f"Muitos NaN no dataset — verifique o período e os ativos."
            )

        logger.debug(
            f"Validação de preços OK: "
            f"{df.shape[1]} ativos, "
            f"fill rate={fill_rate:.1%}, "
            f"range=[{df.index[0].date()}, {df.index[-1].date()}]"
        )

    def _validate_factors(self, df: pd.DataFrame) -> None:
        """Valida fatores NEFIN."""
        expected = {"Rm_minus_Rf", "SMB", "HML", "WML"}
        missing = expected - set(df.columns)
        if missing:
            logger.warning(f"Fatores NEFIN ausentes: {missing}")

        # Fatores diários não devem exceder 30% em valor absoluto
        extreme = (df.abs() > 0.30).any()
        if extreme.any():
            raise ValueError(
                f"Fatores com valores extremos (>30%/dia): "
                f"{df.columns[extreme].tolist()}. "
                f"Verifique se estão em decimal (não em %)."
            )
