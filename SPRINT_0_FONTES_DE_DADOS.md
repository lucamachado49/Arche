# Sprint 0 — Fontes de Dados sem Bloomberg

**Status:** Template — preencher antes do Sprint 1

---

## Contexto

Sem acesso ao Bloomberg Terminal, usamos três fontes gratuitas
em ordem de prioridade:

| Fonte | Cobertura | Ajustado? | Survivorship Bias | Como acessar |
|-------|-----------|-----------|-------------------|--------------|
| **B3 Oficial** | Todos os ativos desde 1994, incluindo deslistados | ❌ Não | ✅ Resolve | Download manual ou via `collector.py` |
| **yfinance** | ~90% dos ativos .SA desde 2000 | ✅ Sim | ❌ Tem (deslistados somem) | `pip install yfinance` |
| **brapi.dev** | >4.000 ativos B3 | ✅ Sim | Parcial | Token gratuito em brapi.dev/dashboard |

---

## Passo a Passo para Obter os Dados

### Opção A — Coleta automática (recomendada)

```bash
# 1. Instalar dependências
pip install yfinance requests pandas pyarrow

# 2. (Opcional) Criar token gratuito em brapi.dev/dashboard
# e exportar como variável de ambiente
export BRAPI_TOKEN="seu_token_aqui"

# 3. Rodar o collector
python -m src.data.collector
# Isso baixa de yfinance + B3 oficial + brapi e salva em:
# data/raw/b3/prices_b3.parquet
```

### Opção B — B3 Oficial manual (mais completo, resolve survivorship bias)

1. Acessar: https://bvmf.bmfbovespa.com.br/InstDados/SerHist/
2. Baixar `COTAHIST_A{ANO}.ZIP` para cada ano de 2012 a 2024
3. Salvar os ZIPs em `data/raw/b3/`
4. O `collector.py` parseia automaticamente

### Opção C — yfinance puro (mais simples, para protótipo rápido)

```python
from src.data.collector import B3DataCollector, IBOVESPA_UNIVERSE
collector = B3DataCollector()
prices = collector.download_yfinance(IBOVESPA_UNIVERSE)
prices.to_parquet("data/raw/b3/prices_b3.parquet")
```

---

## Limitações Conhecidas e Como Lidar

### Survivorship Bias no yfinance
**Problema:** Ativos que faliram ou foram deslistados após 2012 não aparecem.
**Impacto:** O backtest vai superestimar retornos em ~1-3% ao ano.
**Mitigação no projeto:**
- Usar B3 oficial (COTAHIST) como complemento — inclui deslistados
- Documentar a limitação explicitamente no relatório
- A banca entende que dados sem Bloomberg têm essa limitação

### Ajuste por proventos no B3 oficial
**Problema:** COTAHIST não ajusta por dividendos e splits.
**Impacto:** Preços históricos parecem cair nos dias de dividend ex-date.
**Mitigação:**
- Usar yfinance como fonte primária (já ajustado) para ativos cobertos
- B3 oficial apenas como complemento para ativos não cobertos

### Rate limiting do yfinance
**Problema:** Yahoo pode bloquear requisições em massa.
**Mitigação no código:** `batch_size=20` com delay entre batches.
**Se bloquear:** esperar 1 hora ou usar brapi.dev com token.

---

## Decisões da Equipe (preencher)

| Decisão | Escolha | Responsável | Data |
|---------|---------|-------------|------|
| Fonte primária de preços | [ ] yfinance [ ] B3 Oficial [ ] brapi | | |
| Usar brapi.dev? (requer token gratuito) | [ ] Sim [ ] Não | | |
| Baixar B3 Oficial manualmente? | [ ] Sim [ ] Não | | |
| Quem roda o collector.py? | | | |
| Prazo para ter `prices_b3.parquet` pronto | | | |

---

## Gate de Aprovação do Sprint 0

- [ ] `data/raw/b3/prices_b3.parquet` existe com >50 ativos
- [ ] `data/processed/data_quality.csv` gerado pelo collector
- [ ] Fill rate médio >70% no relatório de qualidade
- [ ] NEFIN acessível (download automático ou manual)
- [ ] Toda a equipe entende as limitações documentadas acima
- [ ] `python scripts/check_sprint0.py` retorna status OK

