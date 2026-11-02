# Mega-Sena: dá para vencer o acaso?

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto de portfólio de Data Science sobre o histórico completo de sorteios da Mega-Sena.

O objetivo **não** é prever números. A Mega-Sena é um sorteio físico desenhado para ser
aleatório. A proposta é usar esse histórico para exercitar, de ponta a ponta, um pipeline de
dados com rigor estatístico:

1. **Engenharia de dados**: ETL incremental a partir da API pública da Caixa.
2. **Análise exploratória**: frequências, pares e ímpares, quadrantes do volante.
3. **Inferência**: testes de hipótese corretos para amostragem sem reposição.
4. **Experimentação**: backtest walk-forward que mostra que estratégias de "números quentes"
   ou "números atrasados" performam igual a um gerador pseudoaleatório.

> 📓 **Análise completa, passo a passo:**
> [`notebooks/1.0-lsp-analise-mega-sena.ipynb`](notebooks/1.0-lsp-analise-mega-sena.ipynb)

## Principais resultados

> Esta seção é gerada automaticamente a partir dos resultados (`python -m mega_sena.pipeline`).

<!-- RESULTADOS:INICIO (gerado por mega_sena.report; não edite à mão) -->

Histórico analisado: **3.080 concursos**, de 11/03/1996 a 01/11/2026 (último concurso: 3080).

| Pergunta | Resultado | p-valor |
|---|---|---|
| As 60 dezenas saem com a mesma frequência? | **Não** | 0,007 (qui-quadrado corrigido); 0,008 (Monte Carlo) |
| A proporção de pares segue a teoria? | Sim | 0,198 |
| Os quadrantes recebem 25% cada? | Sim | 0,170 |
| Apostar nas "quentes" vence o acaso? | Não | 0,704 |
| Alguma das 4 estratégias vence o acaso (com Bonferroni)? | Não | menor p: 0,122 |

![Frequência de cada dezena](reports/figures/frequencia_dezenas.png)

**A frequência das dezenas não é perfeitamente uniforme.** O teste global rejeita H0 (p = 0,007), e o teste de Monte Carlo, que não depende de aproximações, confirma (p = 0,008). A dezena 10 saiu 360 vezes e a 26 saiu 250, contra 308 esperadas.

| Período | Concursos | p-valor |
|---|---|---|
| 1996–2005 | 725 | 0,274 |
| 2006–2015 | 1.050 | 0,185 |
| 2016–2026 | 1.305 | < 0,001 |

Essa quebra é **exploratória**: as janelas foram escolhidas depois de ver o teste global, então os p-valores por período não são confirmatórios. O desvio se concentra em 2016–2026 (p < 0,001). Isso sugere hipóteses, como mudança de equipamento ou viés físico das bolas, mas não as confirma: com os dados disponíveis, não dá para separar um viés real de um evento raro do acaso.

![Desvio acumulado](reports/figures/frequencia_acumulada.png)

O desvio acumulado mostra como cada dezena se afasta do esperado ao longo do tempo, com destaque para as 3 mais e as 3 menos sorteadas. O envelope cinza vale para uma dezena isolada: com 60 dezenas, é normal que algumas o ultrapassem.

![Pares e ímpares](reports/figures/pares_impares.png)
![Quadrantes](reports/figures/quadrantes.png)

A composição de pares e ímpares segue a hipergeométrica prevista pela teoria (p = 0,198), e os quadrantes recebem 25% das dezenas cada (p = 0,170).

![Backtest](reports/figures/backtest.png)

| Estratégia | Acertos por concurso | p (Monte Carlo) | p (Bonferroni) |
|---|---|---|---|
| *esperado ao acaso* | *0,600* | | |
| quentes (histórico) | 0,605 | 0,704 | 1,000 |
| quentes (últimos 50) | 0,627 | 0,033 | 0,134 |
| frias (histórico) | 0,572 | 0,030 | 0,122 |
| aleatória | 0,592 | 0,532 | 1,000 |

**Nenhuma estratégia se distingue do acaso** depois da correção para múltiplas comparações. "quentes (últimos 50)" e "frias (histórico)" parecem promissoras isoladamente (p < 0,05), mas foram testadas 4 estratégias: alguma sempre pode parecer boa por sorte, e com Bonferroni a vantagem some. Nenhuma estratégia acertou sequer uma quina em 2.980 apostas.

A lição: um teste estatisticamente significativo não implica um efeito explorável. O tamanho do efeito importa tanto quanto o p-valor.

<!-- RESULTADOS:FIM -->

## Metodologia

### 1. ETL ([`mega_sena/dataset.py`](mega_sena/dataset.py))

- **Fonte**: API pública do Portal de Loterias da Caixa
  (`servicebus2.caixa.gov.br/portaldeloterias/api/megasena/{concurso}`), que retorna um JSON
  por concurso. Como alternativa, `load_results_file()` lê a planilha oficial (CSV/XLSX)
  baixada manualmente do portal.
- **Extract**: o download é incremental e o arquivo bruto (`data/raw/megasena_api.jsonl`) só
  recebe novas linhas. Cada execução baixa só os concursos que faltam, com retry exponencial.
  A API bloqueia rajadas de requisições (HTTP 403), por isso o cliente usa pouco paralelismo e
  faz uma pausa entre as chamadas.
- **Transform**: datas convertidas para `datetime`, dezenas como inteiros, tanto em ordem
  crescente (`d1..d6`) quanto na ordem do sorteio (`bola1..bola6`). A validação checa
  concursos duplicados, dezenas fora de 1..60 e dezenas repetidas dentro de um concurso.
  `to_long()` pivota para o formato longo (uma linha por dezena sorteada).
- **Load**: `data/processed/megasena.csv`, um concurso por linha.

### 2. EDA ([`mega_sena/features.py`](mega_sena/features.py), [`mega_sena/plots.py`](mega_sena/plots.py))

Os **quadrantes** dividem o volante 6×10 em quatro blocos de 15 dezenas:

```
         col 1-5      col 6-10
        +-----------+-----------+
    1-3 |    Q1     |    Q2     |   Q1: 1-5, 11-15, 21-25     Q2: 6-10, 16-20, 26-30
        +-----------+-----------+
    4-6 |    Q3     |    Q4     |   Q3: 31-35, 41-45, 51-55   Q4: 36-40, 46-50, 56-60
        +-----------+-----------+
```

Toda figura traz a referência teórica em cinza, para o desvio observado ser comparado com o que
o acaso produziria, e não com uma expectativa intuitiva.

### 3. Testes de hipótese ([`mega_sena/hypothesis.py`](mega_sena/hypothesis.py))

**H0**: cada concurso é uma amostra uniforme de 6 dezenas distintas entre 60, independente dos
anteriores.

| Teste | O que verifica |
|---|---|
| Qui-quadrado das 60 frequências | Todas as dezenas saem com a mesma probabilidade (6/60) |
| Monte Carlo das 60 frequências | O mesmo, sem depender de aproximação assintótica |
| Qui-quadrado de pares por concurso | Contagem de pares segue a Hipergeométrica(60, 30, 6) |
| Qui-quadrado dos quadrantes | Cada quadrante recebe 25% das dezenas |
| Frequências por década (pós-hoc) | Onde no tempo está o desvio, se houver |

**Detalhe técnico.** O `scipy.stats.chisquare` supõe contagens multinomiais, ou seja, sorteio
*com* reposição. Como as 6 dezenas de um concurso são distintas, a variância real é menor pelo
fator de correção de população finita (60 − 6)/(60 − 1) = 54/59. Por isso a estatística é
dividida por esse fator. O teste [`test_corrected_chi2_is_calibrated_under_h0`](tests/test_hypothesis.py)
simula 400 históricos sob H0 e confirma que o teste corrigido rejeita cerca de 5% das vezes a
5% de significância.

### 4. Backtest ([`mega_sena/modeling/backtest.py`](mega_sena/modeling/backtest.py))

Desenho **walk-forward**: para cada concurso *t*, cada estratégia aposta 6 dezenas olhando
**apenas** os concursos anteriores a *t*. Um teste garante que não há vazamento do futuro.

| Estratégia | Regra |
|---|---|
| quentes (histórico) | as 6 dezenas mais sorteadas até então |
| quentes (últimos 50) | as 6 mais sorteadas nos últimos 50 concursos |
| frias (histórico) | as 6 menos sorteadas ("estão atrasadas") |
| aleatória | gerador pseudoaleatório (`numpy.random.default_rng`) |

Sob H0, os acertos de **qualquer** aposta seguem a Hipergeométrica(60, 6, 6), com média exata de
0,6 por concurso, seja qual for o critério de escolha. Duas formas de testar:

- **Qui-quadrado** da distribuição de acertos (0, 1, 2, 3+) contra a hipergeométrica.
- **Monte Carlo**: 2.000 apostadores aleatórios nos mesmos concursos formam a distribuição nula
  da média de acertos; o p-valor bicaudal mostra onde cada estratégia cai.
- **Bonferroni**: como 4 estratégias são testadas ao mesmo tempo, o p-valor do Monte Carlo é
  multiplicado por 4. Sem isso, a chance de alguma estratégia parecer boa por sorte seria de
  cerca de 19%, e não 5%.

Empates de frequência são desfeitos ao acaso. Desempatar pela menor dezena enviesaria as
apostas para os números baixos.

## Como rodar

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # cria o .venv e instala as dependências

# Atualização completa com os concursos mais recentes (dados, análises, figuras, README e notebook)
uv run python -m mega_sena.pipeline

# Ou cada etapa separadamente:
uv run python -m mega_sena.dataset        # ETL (download incremental)
uv run python -m mega_sena.features       # features por concurso
uv run python -m mega_sena.hypothesis     # testes de hipótese
uv run python -m mega_sena.modeling.backtest
uv run python -m mega_sena.plots          # figuras em reports/figures
uv run python -m mega_sena.report         # seção de resultados do README

uv run pytest                             # testes
```

Com `make` disponível (Linux/macOS/WSL), `make all` faz o mesmo que o pipeline. Todos os passos usam
semente fixa (`RANDOM_SEED` em [`config.py`](mega_sena/config.py)) e são reprodutíveis.

## Organização

```
├── data
│   ├── raw            <- JSONs da API (append-only, fora do git)
│   └── processed      <- megasena.csv e features.csv (fora do git)
├── models             <- Resultados do backtest (resumo, acertos, distribuição nula)
├── notebooks          <- Notebook com a análise completa, já executado
├── reports
│   ├── figures        <- Figuras da EDA e do backtest
│   └── hypothesis_tests.csv
├── tests              <- Testes unitários e de calibração estatística
└── mega_sena
    ├── config.py      <- Caminhos e constantes (60 dezenas, 6 por concurso, semente)
    ├── dataset.py     <- ETL
    ├── features.py    <- Paridade, quadrantes, frequências
    ├── hypothesis.py  <- Testes qui-quadrado (com correção) e Monte Carlo
    ├── plots.py       <- Visualizações
    ├── report.py      <- Textos de resultados gerados a partir dos números (README e notebook)
    ├── pipeline.py    <- Atualização completa em um comando
    └── modeling
        └── backtest.py <- Estratégias e backtest walk-forward
```

Estrutura baseada no [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).
