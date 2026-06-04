# arche/src/__init__.py
# Módulos implementados no Projeto Arche
# Sprint 1 — Dados
from .data import DataLoader, DataCleaner, UniverseBuilder
# Sprint 2 — Grafo
from .graph import GraphBuilder
# Sprint 3 — Curvatura
from .curvature import OllivierRicciCurvature, GDSComputer
# Sprint 4 — Sinal
from .signal import ICAnalyzer, SignalGenerator
# Sprint 5 — Portfólio e Backtest
from .portfolio import PortfolioConstructor, TransactionCostModel
from .backtest import WalkForwardBacktest, PerformanceMetrics
# Sprint 6 — Validação (importação lazy: statsmodels pode ter restrição de DLL no Windows)
try:
    from .validation import FamaFrenchAnalyzer, GrangerCausalityTest, PermutationTest
except ImportError as _e:
    import warnings
    warnings.warn(f"Módulos de validação não disponíveis: {_e}", ImportWarning, stacklevel=2)

