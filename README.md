# Arche — Curvatura de Ricci Discreta como Alpha Cross-Seccional

**Desafio Quant AI 2026 | Itaú Asset Management | FIAP.Quant**

> *"Enquanto modelos tradicionais veem o preço de cada ativo no tempo, o Arche mede a curvatura do espaço onde todos os ativos existem — e a distorção local dessa curvatura é um preditor de alpha que a decomposição de Fama-French confirma como genuíno."*

---

## Hipótese Central

**Existe α > 0 tal que:**

```
E[rᵢ(t+1:t+h) | Δκᵢ(t) < −z·σᵢ] > E[rᵢ(t+1:t+h)] + α
```

Ativos com **queda abrupta na curvatura nodal de Ricci** (GDS — Geometric Divergence Score) estão geometricamente mal posicionados na rede de correlações. Esse mispricing geométrico se corrige nos dias seguintes, gerando alpha mensurável após decomposição Fama-French.

---

## Estrutura do Projeto

```
arche/
├── config/settings.yaml       ← Todos os parâmetros (nunca hardcode)
├── src/
│   ├── data/                  ← Sprint 1: Dados limpos
│   ├── graph/                 ← Sprint 2: MST
│   ├── curvature/             ← Sprint 3: Ollivier-Ricci + GDS
│   ├── signal/                ← Sprint 4: IC e sinal
│   ├── portfolio/             ← Sprint 5: Portfólio long-short
│   ├── backtest/              ← Sprint 5: Walk-forward
│   ├── validation/            ← Sprint 6: Fama-French + Granger
│   ├── transfer/              ← Fase 2: S&P → Ibovespa
│   └── agents/                ← Fase 2: Sistema LLM multiagente
├── tests/
│   └── test_lookahead.py      ← CRÍTICO: deve passar antes de qualquer resultado
├── notebooks/
│   ├── 01_eda_universe.ipynb  ← Validação Sprint 1
│   └── 02_graph_construction.ipynb ← Validação Sprint 2
└── reports/
    └── pre_relatorio/         ← Entregável 31/julho
```

---

## Como Rodar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar dados (Bloomberg)
Exportar preços ajustados da B3 para `data/raw/b3/prices.csv`

Formato esperado:
```
Date,VALE3 BZ Equity,PETR4 BZ Equity,...
2012-01-02,42.50,22.30,...
```

### 3. Rodar testes de look-ahead (OBRIGATÓRIO antes de qualquer resultado)
```bash
pytest tests/test_lookahead.py -v
```

### 4. Sprint 1 — Dados
```bash
jupyter notebook notebooks/01_eda_universe.ipynb
```

### 5. Sprint 2 — Grafo
```bash
jupyter notebook notebooks/02_graph_construction.ipynb
```

---

## Critérios de Avanço por Sprint

| Sprint | Gate de Saída |
|--------|---------------|
| 1 | `test_lookahead.py` passa 100%; fill rate > 70% |
| 2 | Grafo conexo; clusters setoriais visíveis; build < 5s |
| 3 | κᵢ bancos > κᵢ ativos idiossincráticos (sanidade); runtime viável |
| 4 | IC > 0.03 em ≥2 horizontes; IC_IR > 0.30; % positivo > 55% |
| 5 | Sharpe > 0.5 líquido de custos diferenciados |
| 6 | α com t-stat > 2.0 após FF4 NEFIN |

---

## Stack de Papers

| Paper | Papel no Projeto |
|-------|-----------------|
| Ollivier (2009) — Ricci curvature of Markov chains | Base matemática do κ(i,j) |
| Sandhu et al. (2016) — Ricci curvature & systemic risk | Conexão Ricci-finanças |
| Masum & Azad (2026) — Neural Ricci Flow | Fronteira do campo (fev/2026) |
| Poh, Roberts & Zohren (2022) — Transfer Ranking | Fused Encoder Networks |
| Fama & French (1993, 2015) | Validação de alpha |
| Hamilton (1982) + Perelman (2002) | Ricci Flow |

---

## Cronograma

| Data | Entregável |
|------|-----------|
| 31/mai | Inscrição |
| 31/jul | **Pré-Relatório** (Sprints 1–4 completos) |
| 17/ago | **Relatório Final** (Sprints 5–6 + Fase 2 parcial) |
| 26/set | **Final presencial** |
