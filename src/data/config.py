"""
config.py
─────────
Carrega e valida o settings.yaml central.
Todos os módulos importam de cá — nunca carregam o YAML direto.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


# Raiz do projeto (dois níveis acima de src/data/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Carrega o settings.yaml e retorna um dict.
    Cached — carrega apenas uma vez por processo.

    Parameters
    ----------
    path : caminho opcional para o YAML. Se None, usa o padrão.

    Returns
    -------
    dict com todas as configurações.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}\n"
            f"Crie o arquivo ou ajuste o caminho."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _validate_config(cfg)
    logger.info(f"Configuração carregada: {config_path}")
    return cfg


def _validate_config(cfg: dict) -> None:
    """
    Valida campos obrigatórios do settings.yaml.
    Lança ValueError se algo crítico estiver faltando.
    """
    required_sections = [
        "universe", "graph", "curvature",
        "signal", "portfolio", "costs", "backtest",
    ]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(
                f"Seção obrigatória '{section}' ausente no settings.yaml"
            )

    # Validações de domínio
    u = cfg["universe"]
    assert u["min_daily_volume_brl"] > 0, "volume mínimo deve ser positivo"
    assert u["start_date"] < u["end_date"], "start_date deve ser anterior a end_date"

    g = cfg["graph"]
    assert 10 <= g["correlation_window_days"] <= 252, \
        "janela de correlação deve estar entre 10 e 252 dias"
    assert 0.0 <= g["markov_alpha"] <= 1.0, \
        "markov_alpha deve estar em [0, 1]"

    s = cfg["signal"]
    assert 0 < s["ic_min_mean"] < 1, "ic_min_mean deve estar em (0, 1)"
    assert 0.5 < s["ic_min_pct_positive"] < 1, \
        "ic_min_pct_positive deve estar em (0.5, 1)"

    logger.debug("Configuração validada com sucesso.")


def get(section: str, key: str | None = None, default: Any = None) -> Any:
    """
    Atalho para acessar configuração.

    Exemplos
    --------
    get("universe", "min_daily_volume_brl")   → 5_000_000
    get("graph")                               → dict completo da seção
    """
    cfg = load_config()
    if key is None:
        return cfg.get(section, default)
    return cfg.get(section, {}).get(key, default)
