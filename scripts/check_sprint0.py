"""
scripts/check_sprint0.py
────────────────────────
Verifica se o Sprint 0 (fontes de dados) foi concluído
antes de liberar o Sprint 1.

Roda automaticamente no início do notebook 01_eda_universe.ipynb.
Se qualquer check falhar, imprime instruções específicas de correção.

Uso:
    python scripts/check_sprint0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garantir UTF-8 no stdout (necessário no Windows com cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_sprint0() -> bool:
    """
    Retorna True se Sprint 0 está completo, False caso contrário.
    Imprime diagnóstico detalhado em qualquer caso.
    """
    checks = []

    print("=" * 60)
    print("SPRINT 0 — VERIFICAÇÃO DE FONTES DE DADOS")
    print("=" * 60)

    # ── CHECK 1: Arquivo de preços B3 existe ──────────────────
    b3_dir = PROJECT_ROOT / "data" / "raw" / "b3"
    b3_files = (
        list(b3_dir.glob("*.csv")) +
        list(b3_dir.glob("*.parquet")) +
        list(b3_dir.glob("*.xlsx"))
    )
    ok = len(b3_files) > 0
    checks.append(ok)
    status = "✅" if ok else "❌"
    print(f"\n{status} Arquivo de preços B3 em data/raw/b3/")
    if not ok:
        print(
            "   AÇÃO NECESSÁRIA:\n"
            "   1. Abra o Bloomberg Terminal\n"
            "   2. Use BDH ou Excel Add-In para exportar preços\n"
            "   3. Salve como: data/raw/b3/prices_b3.csv\n"
            "   Formato esperado:\n"
            "     Coluna A: Date\n"
            "     Colunas B+: tickers (ex: 'VALE3 BZ Equity')\n"
            "     Valores: PX_LAST ajustado (PX_LAST_ADJ)"
        )
    else:
        print(f"   Encontrado: {[f.name for f in b3_files]}")

    # ── CHECK 2: Lista de tickers existe ──────────────────────
    tickers_file = b3_dir / "tickers_universe.txt"
    ok = tickers_file.exists() and tickers_file.stat().st_size > 0
    checks.append(ok)
    status = "✅" if ok else "⚠️"
    print(f"\n{status} Lista de tickers em data/raw/b3/tickers_universe.txt")
    if not ok:
        print(
            "   AÇÃO NECESSÁRIA (recomendado, não bloqueante):\n"
            "   Crie o arquivo com um ticker por linha:\n"
            "     VALE3 BZ Equity\n"
            "     PETR4 BZ Equity\n"
            "     ...\n"
            "   Se não existir, o loader tentará detectar tickers automaticamente."
        )
        checks[-1] = True  # não bloqueante

    # ── CHECK 3: NEFIN disponível ─────────────────────────────
    nefin_dir = PROJECT_ROOT / "data" / "raw" / "nefin"
    nefin_files = (
        list(nefin_dir.glob("*.csv")) +
        list(nefin_dir.glob("*.xlsx")) +
        list(nefin_dir.glob("*.xls"))
    )
    # NEFIN pode ser baixado automaticamente — não é bloqueante
    ok_nefin = len(nefin_files) > 0
    status = "✅" if ok_nefin else "⚠️"
    print(f"\n{status} Fatores NEFIN em data/raw/nefin/")
    if not ok_nefin:
        print(
            "   Os fatores NEFIN podem ser baixados automaticamente pelo loader.\n"
            "   Se a rede não permitir, baixe manualmente de:\n"
            "     http://nefin.com.br/Risk_Factors/\n"
            "   Arquivos necessários:\n"
            "     Market_Factor.xls, SMB_Factor.xls,\n"
            "     HML_Factor.xls, WML_Factor.xls, Risk_Free.xls\n"
            "   Salve em: data/raw/nefin/"
        )
    else:
        print(f"   Encontrado: {[f.name for f in nefin_files]}")
    checks.append(True)  # não bloqueante

    # ── CHECK 4: settings.yaml existe e é válido ──────────────
    settings_file = PROJECT_ROOT / "config" / "settings.yaml"
    ok = settings_file.exists()
    checks.append(ok)
    status = "✅" if ok else "❌"
    print(f"\n{status} config/settings.yaml")
    if not ok:
        print(
            "   AÇÃO NECESSÁRIA:\n"
            "   O arquivo settings.yaml deve existir em config/.\n"
            "   Ele foi gerado automaticamente pelo projeto.\n"
            "   Verifique se o arquivo não foi deletado."
        )
    else:
        try:
            import yaml
            with open(settings_file) as f:
                cfg = yaml.safe_load(f)
            required = ["universe", "graph", "curvature", "signal"]
            missing = [s for s in required if s not in cfg]
            if missing:
                print(f"   ❌ Seções ausentes no settings.yaml: {missing}")
                checks[-1] = False
            else:
                print(f"   Seções OK: {list(cfg.keys())}")
        except Exception as e:
            print(f"   ❌ Erro ao ler settings.yaml: {e}")
            checks[-1] = False

    # ── CHECK 5: requirements instalados ─────────────────────
    missing_packages = []
    for pkg in ["pandas", "numpy", "networkx", "loguru", "yaml"]:
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
        except ImportError:
            missing_packages.append(pkg)

    ok = len(missing_packages) == 0
    checks.append(ok)
    status = "✅" if ok else "❌"
    print(f"\n{status} Dependências Python instaladas")
    if not ok:
        print(
            f"   AÇÃO NECESSÁRIA:\n"
            f"   pip install -r requirements.txt\n"
            f"   Pacotes faltando: {missing_packages}"
        )

    # ── CHECK 6: Sprint 0 document preenchido ────────────────
    sprint0_doc = PROJECT_ROOT / "SPRINT_0_FONTES_DE_DADOS.md"
    ok = sprint0_doc.exists()
    # Verifica se foi editado (se contém datas preenchidas)
    if ok:
        content = sprint0_doc.read_text()
        has_decisions = "Opção A" in content or "Opção B" in content
        # Não bloqueante — só avisa
        if not has_decisions:
            print(
                f"\n⚠️  SPRINT_0_FONTES_DE_DADOS.md existe mas não foi preenchido.\n"
                f"   Recomendado: preencher antes de começar o Sprint 1."
            )
        else:
            print(f"\n✅ SPRINT_0_FONTES_DE_DADOS.md preenchido")
    checks.append(True)  # não bloqueante

    # ── RESULTADO FINAL ───────────────────────────────────────
    print("\n" + "=" * 60)
    all_pass = all(checks)

    if all_pass:
        print("✅ SPRINT 0 CONCLUÍDO")
        print("   Pipeline liberado. Abra notebooks/01_eda_universe.ipynb")
    else:
        print("❌ SPRINT 0 INCOMPLETO")
        print("   Corrija os itens marcados com ❌ antes de continuar.")
        print("   Itens com ⚠️ são recomendados mas não bloqueantes.")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    ok = check_sprint0()
    sys.exit(0 if ok else 1)
