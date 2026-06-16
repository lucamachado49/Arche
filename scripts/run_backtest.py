"""
scripts/run_backtest.py
───────────────────────
Pipeline completo do backtest walk-forward do Projeto Arche.

Orquestra: preços B3 → GDS (SignalGenerator) → portfólio long-short
quintil → backtest walk-forward → métricas líquidas de custos.

Uso:
    python -m scripts.run_backtest
    python -m scripts.run_backtest --prices data/raw/b3/prices_b3.parquet
    python -m scripts.run_backtest --recompute-gds        # ignora cache de GDS

Por padrão REUTILIZA o cache de GDS já computado em
data/cache/transfer/BR (o mesmo κ/GDS do teste cross-market). Com isso
o backtest roda em segundos. Se o cache não existir, o GDS é computado
do zero (lento: curvatura de Ollivier-Ricci semanal em 2012–2024).

Saídas (em reports/):
    backtest_results.json      — métricas consolidadas (líquido de custos)
    backtest_returns.csv       — série diária de retornos líquidos
    backtest_fold_summary.csv  — métricas por fold (degradação no tempo)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# UTF-8 no stdout (Windows / cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from loguru import logger

from src.data.config import PROJECT_ROOT
from src.data.cleaner import DataCleaner
from src.signal.generator import SignalGenerator
from src.backtest.walkforward import WalkForwardBacktest
from src.backtest.metrics import PerformanceMetrics


# ──────────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS
# ──────────────────────────────────────────────────────────────────
def load_prices(prices_path: Path) -> pd.DataFrame:
    """Carrega preços ajustados da B3 a partir de um parquet."""
    if not prices_path.exists():
        raise FileNotFoundError(
            f"{prices_path} não encontrado.\n"
            f"Rode antes a coleta de dados, por exemplo:\n"
            f"  python -m scripts.collect_cotahist\n"
            f"  python scripts/collect_ibrx100.py"
        )
    prices = pd.read_parquet(prices_path)
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    logger.info(
        f"Preços carregados: {prices.shape[1]} ativos × {len(prices)} dias "
        f"({prices.index[0].date()} → {prices.index[-1].date()})"
    )
    return prices


def load_volumes(volumes_path: Path | None) -> pd.DataFrame | None:
    """Carrega volumes financeiros (opcional). None ativa o proxy de liquidez."""
    if volumes_path is None or not volumes_path.exists():
        if volumes_path is not None:
            logger.warning(f"Volumes {volumes_path} não encontrado — usando proxy de liquidez.")
        return None
    volumes = pd.read_parquet(volumes_path)
    if not isinstance(volumes.index, pd.DatetimeIndex):
        volumes.index = pd.to_datetime(volumes.index)
    logger.info(f"Volumes carregados: {volumes.shape[1]} ativos × {len(volumes)} dias")
    return volumes.sort_index()


def try_load_risk_free() -> pd.Series | None:
    """Tenta carregar a Selic diária (risk-free) do BCB. Falha silenciosa."""
    try:
        from src.data.loader import DataLoader
        rf = DataLoader().load_selic(download_if_missing=True)
        logger.info(f"Risk-free (Selic) carregado: {len(rf)} dias")
        return rf
    except Exception as e:
        logger.warning(f"Selic indisponível ({e}) — Sharpe sem risk-free explícito.")
        return None


# ──────────────────────────────────────────────────────────────────
# RELATÓRIO
# ──────────────────────────────────────────────────────────────────
def raw_metrics(pm: PerformanceMetrics) -> dict:
    """Métricas numéricas (não formatadas) para serialização JSON."""
    out = {
        "annualized_return": pm.annualized_return(),
        "annualized_vol":    pm.annualized_vol(),
        "sharpe_ratio":      pm.sharpe_ratio(),
        "sortino_ratio":     pm.sortino_ratio(),
        "calmar_ratio":      pm.calmar_ratio(),
        "max_drawdown":      pm.max_drawdown(),
        "win_rate":          pm.win_rate(),
        "var_95":            pm.var_95(),
        "cvar_95":           pm.cvar_95(),
        "skewness":          pm.skewness(),
        "kurtosis":          pm.kurtosis(),
        "t_stat_return":     pm.t_stat_return(),
        "n_obs":             int(len(pm.returns)),
        "start":             str(pm.returns.index[0].date()),
        "end":               str(pm.returns.index[-1].date()),
    }
    ir = pm.information_ratio()
    if ir is not None:
        out["information_ratio"] = ir
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest walk-forward GDS (Arche)")
    parser.add_argument(
        "--prices",
        default=str(PROJECT_ROOT / "data" / "raw" / "b3" / "prices_b3.parquet"),
        help="Parquet de preços ajustados da B3.",
    )
    parser.add_argument(
        "--volumes",
        default=None,
        help="Parquet de volumes financeiros (opcional). Se omitido, usa proxy de liquidez.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "data" / "cache" / "transfer" / "BR"),
        help="Diretório de cache do GDS. Por padrão reutiliza o cache do teste cross-market.",
    )
    parser.add_argument(
        "--recompute-gds",
        action="store_true",
        help="Ignora o cache e recomputa a curvatura/GDS do zero (lento).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Paralelismo no cálculo da curvatura (só quando recomputa).",
    )
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "reports"),
        help="Diretório de saída dos relatórios.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Dados ──────────────────────────────────────────────────
    prices = load_prices(Path(args.prices))
    volumes = load_volumes(Path(args.volumes) if args.volumes else None)

    # ── 2. GDS (reaproveita cache se existir) ─────────────────────
    use_cache = not args.recompute_gds
    cache_dir = Path(args.cache_dir)
    logger.info(
        f"Gerando GDS (cache={'on' if use_cache else 'off'}, dir={cache_dir})..."
    )
    generator = SignalGenerator(cache_dir=cache_dir)
    _, gds_ts = generator.run(
        prices=prices,
        volumes=volumes,
        n_jobs=args.n_jobs,
        use_cache=use_cache,
    )

    # Alinhar preços ao universo do GDS (o cache pode ter menos ativos)
    common = [c for c in gds_ts.columns if c in prices.columns]
    prices = prices[common]
    logger.info(f"GDS: {gds_ts.shape} | universo de backtest: {len(common)} ativos")

    # ── 3. Retornos (log, consistente com o resto do pipeline) ────
    returns = DataCleaner().prices_to_log_returns(prices)

    # Benchmark proxy: equal-weight do universo (para Information Ratio)
    benchmark = returns.mean(axis=1)
    benchmark.name = "EW_universe"

    # Risk-free (Selic) — opcional
    risk_free = try_load_risk_free()

    # ── 4. Backtest walk-forward ──────────────────────────────────
    engine = WalkForwardBacktest()
    folds = engine.run(
        returns=returns,
        gds_ts=gds_ts,
        prices=prices,
        benchmark=benchmark,
        risk_free=risk_free,
    )

    if not folds:
        logger.error("Nenhum fold concluído — verifique o período e o GDS.")
        sys.exit(1)

    # ── 5. Consolidação ───────────────────────────────────────────
    net_returns = engine.get_combined_returns(folds, net=True)
    gross_returns = engine.get_combined_returns(folds, net=False)

    pm_net = PerformanceMetrics(net_returns, benchmark=benchmark, risk_free=risk_free)
    pm_gross = PerformanceMetrics(gross_returns, benchmark=benchmark, risk_free=risk_free)
    fold_summary = engine.get_fold_summary(folds)

    # ── 6. Saída ──────────────────────────────────────────────────
    results = {
        "config": {
            "prices_file": str(Path(args.prices).name),
            "n_assets": len(common),
            "n_folds": len(folds),
            "train_years": engine.train_years,
            "test_months": engine.test_months,
            "step_months": engine.step_months,
            "oos_start": str(engine.oos_start.date()),
        },
        "net": raw_metrics(pm_net),
        "gross": raw_metrics(pm_gross),
    }

    json_path = out_dir / "backtest_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    net_returns.to_csv(out_dir / "backtest_returns.csv", header=True)
    fold_summary.to_csv(out_dir / "backtest_fold_summary.csv")

    # ── 7. Console ────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("BACKTEST WALK-FORWARD — PORTFÓLIO GDS LONG-SHORT (líquido)")
    print("═" * 62)
    print(pm_net.summary().to_string())
    print("\n── Métricas por fold ──")
    print(fold_summary.to_string())
    print("\n── Bruto vs Líquido ──")
    print(f"  Sharpe bruto   : {pm_gross.sharpe_ratio():.4f}")
    print(f"  Sharpe líquido : {pm_net.sharpe_ratio():.4f}")
    print(f"\nRelatórios salvos em: {out_dir}")
    print(f"  • {json_path.name}")
    print(f"  • backtest_returns.csv")
    print(f"  • backtest_fold_summary.csv")


if __name__ == "__main__":
    main()
