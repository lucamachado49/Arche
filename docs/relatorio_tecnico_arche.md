# Arche — Relatório Técnico

### A curvatura de Ricci carrega alpha explorável na B3? Um estudo de validação com falha honesta

**Desafio Quant AI 2026 · Itaú Asset Management**
*Robô: **Arche** — "Antes do preço, a geometria."*

---

## Resumo

Testamos a hipótese de que a **geometria da rede de correlações** da B3 — medida pela
**curvatura de Ollivier-Ricci** — carrega informação preditiva cross-seccional de retornos.
Construímos o pipeline completo (grafo → curvatura → sinal GDS → portfólio long-short →
backtest walk-forward líquido de custos) e submetemos o sinal à mesma bateria de ceticismo
de uma mesa institucional.

**Conclusão honesta:** o sinal geométrico é **genuíno e ~95% ortogonal** aos fatores de risco
conhecidos (mercado, reversão, momentum, baixa volatilidade), e — com a construção correta —
tem **IC cross-seccional estatisticamente significativo e robusto a excluir 2020** (t = 2,2).
**Porém**, sua magnitude (IC ≈ 0,013) está **abaixo do limiar de lucratividade líquida de
custos** da B3, e ~todo o retorno bruto histórico está **concentrado no regime de 2020**.
Não é um produto explorável isolado; é um **resultado de pesquisa condicional/negativo, obtido
com rigor**. O valor entregue é a metodologia — não um Sharpe inflado.

---

## 1. Hipótese

O mercado não é um conjunto de séries temporais isoladas: é uma **rede de correlações** com
forma e curvatura. Ativos com **queda abrupta na curvatura nodal** estão geometricamente mal
posicionados na rede — um mispricing que tenderia a corrigir.

> **H₀:** E[ r_i(t+1:t+h) | ΔκᵢΔ(t) anômalo ] ≠ E[ r_i(t+1:t+h) ], com α > 0 após controlar fatores.

## 2. Dados e método

| Item | Escolha |
|---|---|
| Universo | Ações líquidas B3 (base IBrX-100), 87 ativos, volume > R$5 mi/dia |
| Período | 2013–2024 (cache de curvatura semanal), OOS a partir de 2019 |
| Grafo | Janela 60d → correlação → distância √(2(1−ρ)) → MST + arestas fortes |
| Curvatura | Ollivier-Ricci por aresta (transporte ótimo / EMD); κᵢ = média nodal |
| Sinal | GDS = variação padronizada de κᵢ (z-score) |
| Portfólio | Ranking cross-seccional → quintil long-short market-neutral |
| Backtest | Walk-forward, treino 3a / teste 6m / passo 3m; **líquido de custos B3** |
| Custos | 13–22 bps round-trip (por liquidez) + ~3% a.a. de aluguel no short |
| Anti-viés | Look-ahead gate (teste dedicado), point-in-time, sem survivorship |

## 3. Protocolo de testes (a bateria de ceticismo)

1. **Testes de look-ahead** (gate obrigatório) — 34 testes unitários, 100% verdes.
2. **IC cross-seccional** por horizonte (h = 1, 5, 10, 21) com permutação.
3. **Backtest walk-forward** bruto vs. líquido de custos.
4. **Decomposição temporal** (ano a ano; com/sem 2020).
5. **Ortogonalidade a fatores** (Fama-French caseiro: MKT, STR, MOM, LOWV; HAC).
6. **Métodos de construção** (quintil vs. z-score vs. mean-variance Σ⁻¹).
7. **Redução de turnover** (banda de no-trade; hysteresis de ranking).
8. **Condicionamento por regime** (dispersão de correlações, point-in-time).
9. **Pivô risk-monitor** (curvatura agregada vs. vol/drawdown futuros; incremental à vol).
10. **Reconstrução do sinal** (Δκ de 1 vs. 4, 8, 12 semanas; nível vs. tendência).

## 4. Resultados — a jornada de reconciliação

A primeira coisa que a auditoria encontrou foi que o resultado inicial estava **corrompido por
bugs**. Corrigi-los mudou o quadro por completo:

| Versão do modelo | Sharpe bruto | Sharpe líquido | Ret. líq. a.a. | MaxDD líq. |
|---|---|---|---|---|
| Inicial (commitado) — sinal invertido + semanal | −0,66 | −3,93 | −46,8% | −97,6% |
| Correção do sinal + rebalance mensal | +0,22 | −0,63 | −8,9% | −48,0% |
| + Suavização EWMA (sinal lento, ~1 trimestre) | +0,48 | −0,18 | −3,8% | −42,5% |
| **+ Engine contínuo (final)** | **+0,49** | **−0,15** | **−3,7%** | **−47%** |

**Métrica final (OOS 2019–2024):** bruto +6,75% a.a. (Sharpe 0,486, t = 1,18); líquido −3,7% a.a.

## 5. Onde engasgamos (os pontos de falha)

1. **Sinal invertido.** O portfólio comprava o quintil de *menor* retorno esperado — o IC
   realizado é positivo, mas o book estava do lado errado. Correção levou o bruto de −0,66 a +0,22.
2. **Turnover come o alpha.** Rebalance semanal de um sinal mensal quadruplicava o custo. O
   líquido só deixou de ser catastrófico ao casar a frequência com o horizonte do sinal (h≈21).
3. **Construções "sofisticadas" pioram.** Espalhar peso por todo o cross-section (z-score,
   bruto −0,19) ou rotacionar por risco (mean-variance Σ⁻¹, bruto −0,50) **destroem** o sinal —
   o alpha é **concentrado nas caudas**. O quintil simples é o melhor estimador.
4. **Banda de no-trade e hysteresis: rejeitadas.** A banda é inócua (turnover vem de rotação de
   cauda, não de micro-oscilação); a hysteresis corta turnover mas derruba o bruto (0,49 → 0,17).
5. **Magnitude é a restrição vinculante.** Nenhuma engenharia de portfólio fura um IC de ~0,013
   contra os custos da B3.

## 6. A aparada do COVID (o teste que definiu o veredito)

Suspeitamos que um sinal de geometria de correlações teria todo o payoff no choque de 2020.
Decompusemos o retorno bruto ano a ano:

| Ano | Ret. bruto a.a. | t |
|---|---|---|
| 2013–2017 | **todos negativos** (−1,5% a −17,6%) | — |
| 2018 | +18,2% | 1,17 |
| 2019 | +19,1% | 1,51 |
| **2020** | **+35,3%** | 1,38 |
| 2021 | +4,2% | 0,28 |
| 2022 | −16,9% | −1,19 |
| 2023 | +7,3% | 0,53 |
| 2024 | −3,4% | −0,30 |

E o corte com/sem 2020:

| Recorte | Ret. a.a. | t |
|---|---|---|
| Amostra cheia | +2,52% | 0,52 |
| **Sem 2020** | **−0,60%** | **−0,13** |
| OOS 2019+ | +7,57% | 1,15 |
| **OOS sem 2020** | **+2,01%** | **0,33** |

**2020 sozinho responde por 122% do retorno acumulado vitalício.** Sem 2020, a estratégia é
flat-a-negativa. E ela **falhou nas outras deslocações** (pior ano = 2015, crise Brasil, −17,6%;
2022, −16,9%) — logo, não é nem um hedge de cauda confiável. Verificamos que a *breadth* de
2014–2017 era saudável (~57–61 nomes): os anos negativos são **reais, não ruído de universo fino**.

## 7. Ortogonalidade — genuíno, mas fraco

Regressão do L/S bruto contra fatores caseiros (MKT, STR, MOM, LOWV; erros HAC):

| Amostra | α ajustado (a.a.) | t_NW | R² |
|---|---|---|---|
| OOS 2019–24 | +9,36% | 1,29 | 0,069 |
| Cheia 2013–24 | +3,27% | 0,68 | 0,041 |

**Boa notícia:** R² < 7% → o sinal é **~95% ortogonal** aos fatores conhecidos; **não é reversão
reembalada**. **Má notícia:** o α é economicamente relevante mas **estatisticamente insignificante**
(t ≤ 1,3) — e some sem 2020.

## 8. Reconstrução do sinal — melhora o IC, não a economia

O GDS original media Δκ de **1 semana** (ruído de rewiring da MST). Testamos horizontes maiores:

| Definição do sinal | IC | t | IC ex-2020 | t ex-2020 |
|---|---|---|---|---|
| Δκ 1 semana (atual) | +0,0106 | 1,83 | +0,0072 | 1,20 |
| **Δκ 4 semanas** (≈ h*=21) | **+0,0176** | **3,02** | **+0,0129** | **2,22** |
| Δκ 8 / 12 semanas | fracos / negativos | — | — | — |

O **Δκ de 4 semanas** quase dobra o IC e é a **única variante cujo IC ex-2020 é significativo
(t = 2,2)** — um achado de pesquisa real. **Porém**, líquido de custo o retorno **não melhora**
(o ganho de IC é minúsculo e vive no *meio* da distribuição, não nas caudas negociadas).

## 9. Ricci como monitor de risco? (o pivô, também refutado)

Testamos a curvatura **agregada** (κ̄ médio do mercado) como preditor de vol/drawdown futuros.

- κ̄ prevê vol futura: Spearman **ρ = +0,33 (p < 0,001)** — parece forte.
- **Teste incremental** (fwd_vol ~ vol_atual + κ̄): **t_κ̄ = −0,48 (p = 0,63)** → controlando pela
  vol corrente, a curvatura **não adiciona nada**. É um proxy (caro) de volatilidade realizada.
- Overlay de risco com curvatura (Sharpe 0,26; MaxDD −45%) **perde** para um vol-targeting trivial
  (Sharpe 0,38; MaxDD −44%; e ainda com retorno maior).

## 10. Por que "não passou" — os gates do projeto

| Gate (pré-definido) | Alvo | Resultado | Status |
|---|---|---|---|
| Sprint 1 — look-ahead | 100% | 34/34 testes | ✅ |
| Sprint 4 — IC | > 0,03 em ≥2 horizontes | 0,047 em h=21 (só 1 horizonte robusto) | ⚠️ parcial |
| Sprint 5 — Sharpe líquido | > 0,5 | −0,15 | ❌ |
| Sprint 6 — α pós-fatores | t > 2,0 | t = 0,68–1,29 | ❌ |

O modelo **funciona** (roda ponta-a-ponta, é look-ahead-safe, honesto), mas **não atinge os gates
de lucratividade/significância** — porque o sinal, embora real e ortogonal, é fraco e concentrado
em um regime.

## 11. Cronograma de desenvolvimento e esforço

> *Datas de calendário a preencher pela equipe; abaixo, as fases e os artefatos entregues.*

| Fase | Escopo | Artefatos |
|---|---|---|
| 0–1 · Dados | Universo B3, limpeza, filtro de liquidez, testes de look-ahead | `src/data/`, `tests/test_lookahead.py` |
| 2 · Grafo | MST + arestas fortes, clusters setoriais | `src/graph/builder.py`, notebook 02 |
| 3 · Curvatura | Ollivier-Ricci (EMD), curvatura nodal κᵢ, sanidade (bancos > idioss.) | `src/curvature/` |
| 4 · Sinal | GDS, IC por horizonte, permutação | `src/signal/`, `run_transfer_test` |
| 5 · Portfólio | Quintil L/S, custos diferenciados, walk-forward | `src/portfolio/`, `src/backtest/`, `run_backtest` |
| 6 · **Validação (auditoria de rigor)** | **9 bancadas de diagnóstico; dezenas de configurações de backtest** | `scripts/diag_*.py` |

A fase de **validação** foi a mais intensa: correção de bugs metodológicos, reconstrução do
engine (versão contínua sem artefato de recompra por fold), e **9 experimentos independentes**
(concentração em 2020, ortogonalidade, regime, custo, reconstrução do sinal, monitor de risco).
Todos reprodutíveis via `python -m scripts.diag_*`.

## 12. Conclusão

A curvatura de Ollivier-Ricci no grafo de correlações da B3, **neste dataset**, carrega informação
cross-seccional **genuína e ortogonal** aos fatores conhecidos, porém **estatisticamente fraca e
concentrada no regime de 2020** — e **não explorável líquida de custos** em nenhuma estrutura
testada (long-short, long-only enhanced index, condicional). Como monitor de risco agregado, é
**dominada pela volatilidade realizada**.

Isto não é um fracasso de execução — é o **resultado correto**, alcançado com o mesmo protocolo
de ceticismo de uma mesa institucional: encontramos um sinal, derrubamos um Sharpe inflado
(artefato de subamostragem), corrigimos um sinal invertido, separamos custo de sinal, provamos
ortogonalidade e atribuímos honestamente o retorno ao seu verdadeiro driver.

## 13. Próximos passos

1. **Dados** — pesos float-cap reais do IBrX-100 (torna o lado short do enhanced index investível)
   e universo ampliado (~250 nomes) para mais poder cross-seccional.
2. **Sinal** — features geométricas além do Δκ de 1 semana: Forman-Ricci, fluxo de Ricci,
   curvatura de nível vs. tendência lenta.
3. **Validação oficial** — decomposição Fama-French 4 fatores com dados NEFIN/FEA-USP.

---

## Apêndice — Reprodutibilidade

```bash
pip install -r requirements.txt
pytest                                 # 34 testes; gate de look-ahead
python -m scripts.run_backtest         # pipeline completo → reports/
python -m scripts.diag_covid           # a aparada do COVID (Seção 6)
python -m scripts.diag_orthogonality   # ortogonalidade a fatores (Seção 7)
python -m scripts.diag_signal_recon    # reconstrução do sinal (Seção 8)
python -m scripts.diag_stress          # monitor de risco (Seção 9)
```

*Todas as métricas deste relatório são reproduzíveis a partir do cache de curvatura em
`data/cache/transfer/BR/` e dos scripts em `scripts/diag_*.py`.*
