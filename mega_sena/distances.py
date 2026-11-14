"""Distância entre as dezenas de um mesmo concurso.

Perguntas respondidas:

* Com que frequência saem dezenas **consecutivas** (ex.: 22 e 23) no mesmo concurso?
* Qual a **menor distância** entre duas dezenas de um concurso, e como se distribuem os saltos
  entre dezenas vizinhas na ordem crescente?
* **Quando sai a dezena x**, a que distância ficam as outras? A vizinha mais próxima fica mais
  perto ou mais longe do que o acaso prevê?

Sob H0 (6 dezenas distintas e uniformes entre 60), todas essas distribuições têm fórmula exata,
obtida por contagem de combinações. As fórmulas estão nas funções ``*_pmf`` e são conferidas
contra simulação nos testes automatizados.

Uso::

    python -m mega_sena.distances   # grava reports/distance_tests.csv
"""

from dataclasses import asdict
from functools import cache
from math import comb
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mega_sena.config import (
    DISTANCE_TESTS_FILE,
    DRAW_COLUMNS,
    NUMBERS_PER_DRAW,
    PROCESSED_RESULTS_FILE,
    RANDOM_SEED,
    TOTAL_NUMBERS,
)
from mega_sena.dataset import load_processed
from mega_sena.hypothesis import TestResult, pmf_test, random_draws

app = typer.Typer()

N, K = TOTAL_NUMBERS, NUMBERS_PER_DRAW
GAP_CAP = 25  # saltos >= 25 ficam numa categoria só (são raros)

# Nomes dos testes na tabela de resultados (também usados por mega_sena.report)
CONSECUTIVE_TEST = "Pares consecutivos por concurso"
LONGEST_RUN_TEST = "Maior sequência de consecutivas por concurso"
MIN_GAP_TEST = "Menor distância por concurso"
GAP_TEST = "Saltos entre dezenas vizinhas (Monte Carlo)"
PAIR_TEST = "Distância entre todos os pares (Monte Carlo)"
NEAREST_TEST = "Vizinha mais próxima de cada dezena (Monte Carlo)"

_PAIR_I, _PAIR_J = np.triu_indices(K, k=1)

# Graus de liberdade de referência dos testes de Monte Carlo (o p-valor não depende deles)
_MC_DOF = {GAP_TEST: GAP_CAP - 1, PAIR_TEST: N - 2, NEAREST_TEST: N}


# ---------------------------------------------------------------------------------------------
# Métricas observadas
# ---------------------------------------------------------------------------------------------


def sorted_gaps(draws: np.ndarray) -> np.ndarray:
    """Saltos entre dezenas vizinhas na ordem crescente: matriz (concursos, 5).

    Ex.: o concurso 04 05 30 33 41 52 tem saltos 1, 25, 3, 8, 11.
    """
    return np.diff(np.sort(draws, axis=1), axis=1)


def consecutive_pairs(draws: np.ndarray) -> np.ndarray:
    """Quantos pares de dezenas consecutivas (salto 1) cada concurso tem."""
    return (sorted_gaps(draws) == 1).sum(axis=1)


def longest_run(draws: np.ndarray) -> np.ndarray:
    """Maior sequência de dezenas consecutivas em cada concurso (1 = nenhuma consecutiva).

    Ex.: 01 02 03 10 20 60 tem sequência 3 (01-02-03).
    """
    is_step = sorted_gaps(draws) == 1
    best = np.ones(len(draws), dtype=int)
    current = np.ones(len(draws), dtype=int)
    for column in is_step.T:
        current = np.where(column, current + 1, 1)
        best = np.maximum(best, current)
    return best


def min_gap(draws: np.ndarray) -> np.ndarray:
    """Menor distância entre duas dezenas de cada concurso (1 = há consecutivas)."""
    return sorted_gaps(draws).min(axis=1)


def pair_distances(draws: np.ndarray) -> np.ndarray:
    """Distância entre cada um dos 15 pares de dezenas de cada concurso: matriz (concursos, 15)."""
    return np.abs(draws[:, _PAIR_I] - draws[:, _PAIR_J])


def nearest_neighbor(draws: np.ndarray) -> np.ndarray:
    """Para cada dezena sorteada, a distância até a outra dezena mais próxima do mesmo concurso.

    Returns:
        Matriz (concursos, 6), alinhada com ``draws``.
    """
    diff = np.abs(draws[:, :, None] - draws[:, None, :]).astype(float)
    diag = np.arange(draws.shape[1])
    diff[:, diag, diag] = np.inf
    return diff.min(axis=2).astype(int)


def neighbor_offsets(draws: np.ndarray, number: int) -> np.ndarray:
    """Posição relativa (``y - number``) das outras dezenas nos concursos em que ``number`` saiu.

    Ex.: se sai a 10 junto com a 12 e a 7, os deslocamentos incluem +2 e -3.
    """
    rows = draws[(draws == number).any(axis=1)]
    others = rows[rows != number]
    return others - number


def nearest_by_number(draws: np.ndarray) -> pd.DataFrame:
    """Distância média até a vizinha mais próxima, por dezena, comparada ao esperado.

    Returns:
        Uma linha por dezena: ``vezes`` (quantas vezes saiu), ``media`` observada,
        ``esperado`` e ``desvio_padrao`` teóricos de uma ocorrência, e ``z`` (desvio da média
        em erros-padrão).
    """
    numbers = draws.ravel()
    counts = np.bincount(numbers, minlength=N + 1)[1:]
    sums = np.bincount(numbers, weights=nearest_neighbor(draws).ravel(), minlength=N + 1)[1:]
    table = nearest_theory().copy()
    table.insert(0, "vezes", counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        table.insert(1, "media", sums / counts)
        table["z"] = (table["media"] - table["esperado"]) / (
            table["desvio_padrao"] / np.sqrt(table["vezes"])
        )
    return table


# ---------------------------------------------------------------------------------------------
# Distribuições teóricas (exatas) sob H0
# ---------------------------------------------------------------------------------------------


def _mean_sd(pmf: pd.Series) -> tuple[float, float]:
    values = pmf.index.to_numpy(dtype=float)
    mean = float((values * pmf).sum())
    return mean, float(np.sqrt(((values - mean) ** 2 * pmf).sum()))


@cache
def nearest_theory() -> pd.DataFrame:
    """Média e desvio-padrão teóricos da distância até a vizinha mais próxima, por dezena.

    As dezenas das pontas (1, 60) têm vizinhas só de um lado, por isso a distância esperada é
    maior para elas.
    """
    return pd.DataFrame(
        [_mean_sd(nearest_neighbor_pmf(x)) for x in range(1, N + 1)],
        index=pd.RangeIndex(1, N + 1, name="dezena"),
        columns=["esperado", "desvio_padrao"],
    )


def consecutive_pairs_pmf() -> pd.Series:
    """P(o concurso ter exatamente j pares consecutivos), j = 0..5.

    Fórmula clássica de "sucessões": C(K-1, j) * C(N-K+1, K-j) / C(N, K). Para j = 0 dá
    C(55, 6) / C(60, 6) ≈ 0,579, ou seja, cerca de 42% dos concursos têm ao menos um par de
    dezenas consecutivas.
    """
    pmf = [comb(K - 1, j) * comb(N - K + 1, K - j) / comb(N, K) for j in range(K)]
    return pd.Series(pmf, index=pd.RangeIndex(K, name="pares_consecutivos"))


def _count_max_run_at_most(r: int) -> int:
    """Quantas combinações de K dezenas entre N não têm sequência de consecutivas maior que r.

    Programação dinâmica percorrendo o volante de 1 a N: o estado é (dezenas já escolhidas,
    tamanho da sequência atual terminando na posição corrente).
    """
    ways = np.zeros((K + 1, r + 1), dtype=object)  # object: inteiros exatos, sem overflow
    ways[0, 0] = 1
    for _ in range(N):
        new = np.zeros_like(ways)
        new[:, 0] = ways.sum(axis=1)  # posição não escolhida: a sequência zera
        new[1:, 1:] = ways[:-1, :-1]  # posição escolhida: a sequência cresce
        ways = new
    return int(ways[K].sum())


def longest_run_pmf() -> pd.Series:
    """P(maior sequência de dezenas consecutivas do concurso = r), r = 1..6 (exata)."""
    at_most = [_count_max_run_at_most(r) for r in range(K + 1)]
    pmf = np.diff(at_most) / comb(N, K)
    return pd.Series(pmf, index=pd.RangeIndex(1, K + 1, name="maior_sequencia"))


def min_gap_pmf() -> pd.Series:
    """P(menor distância do concurso = g).

    Subtraindo (g-1)·i da i-ésima menor dezena, os concursos com todos os saltos >= g viram
    combinações quaisquer de N - (g-1)(K-1) números: P(menor >= g) = C(N - (g-1)(K-1), K) / C(N, K).
    """
    survival = []
    g = 1
    while (remaining := N - (g - 1) * (K - 1)) >= K:
        survival.append(comb(remaining, K) / comb(N, K))
        g += 1
    survival.append(0.0)
    pmf = -np.diff(survival)
    return pd.Series(pmf, index=pd.RangeIndex(1, len(pmf) + 1, name="menor_distancia"))


def gap_pmf(cap: int = GAP_CAP) -> pd.Series:
    """P(salto entre duas dezenas vizinhas na ordem crescente = g), igual para os 5 saltos.

    Juntar duas vizinhas a distância g num só bloco deixa C(N - g, K - 1) maneiras de escolher o
    resto, logo P(salto = g) = C(N - g, K - 1) / C(N, K). Saltos >= ``cap`` são somados na
    última categoria.
    """
    pmf = np.array([comb(N - g, K - 1) / comb(N, K) for g in range(1, N - K + 2)])
    capped = np.append(pmf[: cap - 1], pmf[cap - 1 :].sum())
    return pd.Series(capped, index=pd.RangeIndex(1, cap + 1, name="salto"))


def pair_distance_pmf() -> pd.Series:
    """P(distância entre duas dezenas de um concurso = d) = (N - d) / C(N, 2), d = 1..59.

    Há N - d pares de dezenas a distância d no volante, e todos os C(N, 2) pares são igualmente
    prováveis. Distâncias curtas são mais comuns só porque existem mais pares assim.
    """
    d = np.arange(1, N)
    return pd.Series((N - d) / comb(N, 2), index=pd.RangeIndex(1, N, name="distancia"))


def nearest_neighbor_pmf(number: int) -> pd.Series:
    """P(a vizinha mais próxima da dezena ``number`` estar a distância d), dado que ela saiu.

    As outras 5 dezenas são uma amostra uniforme das 59 restantes. Se ``m(d)`` dezenas ficam a
    menos de d de ``number`` (menos nas pontas do volante, 1 e 60), então
    P(vizinha >= d) = C(59 - m(d), 5) / C(59, 5).
    """
    others = N - 1
    survival = []
    for d in range(1, N + 1):
        within = min(d - 1, number - 1) + min(d - 1, N - number)
        survival.append(comb(others - within, K - 1) / comb(others, K - 1))
    pmf = -np.diff(survival)
    return pd.Series(pmf, index=pd.RangeIndex(1, N, name="distancia"))


# ---------------------------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------------------------


def _chi2(counts: np.ndarray, expected: np.ndarray) -> float:
    return float(((counts - expected) ** 2 / expected).sum())


def _dependent_statistics(draws: np.ndarray) -> dict[str, float]:
    """Estatísticas cujas observações não são independentes dentro de um concurso.

    Os 5 saltos, os 15 pares e as 6 vizinhas de um mesmo concurso são correlacionados, então o
    qui-quadrado não segue a distribuição teórica; o p-valor vem de simulação (Monte Carlo).
    """
    n = len(draws)
    gaps = np.minimum(sorted_gaps(draws), GAP_CAP).ravel()
    pairs = pair_distances(draws).ravel()
    nearest = nearest_by_number(draws)
    return {
        GAP_TEST: _chi2(
            np.bincount(gaps, minlength=GAP_CAP + 1)[1:], gap_pmf().to_numpy() * n * (K - 1)
        ),
        PAIR_TEST: _chi2(
            np.bincount(pairs, minlength=N)[1:], pair_distance_pmf().to_numpy() * n * len(_PAIR_I)
        ),
        NEAREST_TEST: float((nearest["z"].dropna() ** 2).sum()),
    }


def run_distance_tests(
    df: pd.DataFrame, n_sims: int = 2000, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Testa se as distâncias entre as dezenas seguem o esperado num sorteio uniforme.

    Os três primeiros testes usam uma observação por concurso (independentes) e o qui-quadrado
    tradicional. Os demais usam várias observações correlacionadas por concurso e p-valor de
    Monte Carlo. Como são vários testes, a coluna ``p_valor_bonferroni`` corrige para múltiplas
    comparações.

    Returns:
        Uma linha por teste.
    """
    draws = df[DRAW_COLUMNS].to_numpy(dtype=int)
    results = [
        pmf_test(
            np.bincount(consecutive_pairs(draws), minlength=K),
            consecutive_pairs_pmf(),
            CONSECUTIVE_TEST,
        ),
        pmf_test(
            np.bincount(longest_run(draws), minlength=K + 1)[1:],
            longest_run_pmf(),
            LONGEST_RUN_TEST,
        ),
        pmf_test(
            np.bincount(min_gap(draws), minlength=len(min_gap_pmf()) + 1)[1:],
            min_gap_pmf(),
            MIN_GAP_TEST,
        ),
    ]

    rng = np.random.default_rng(seed)
    observed = _dependent_statistics(draws)
    simulated = [_dependent_statistics(random_draws(len(draws), rng)) for _ in range(n_sims)]
    for name, value in observed.items():
        null = np.array([sim[name] for sim in simulated])
        p_value = (1 + (null >= value).sum()) / (n_sims + 1)
        results.append(TestResult(name, value, _MC_DOF[name], p_value, f"{n_sims} simulações"))

    table = pd.DataFrame([asdict(result) for result in results])
    table["p_valor_bonferroni"] = (table["p_valor"] * len(table)).clip(upper=1.0)
    return table


@app.command()
def main(
    input_path: Path = PROCESSED_RESULTS_FILE,
    output_path: Path = DISTANCE_TESTS_FILE,
    n_sims: int = typer.Option(2000, help="Simulações dos testes de Monte Carlo."),
    seed: int = RANDOM_SEED,
):
    """Roda os testes de distância entre as dezenas e salva a tabela de resultados."""
    logger.info("Rodando testes de distância entre as dezenas...")
    results = run_distance_tests(load_processed(input_path), n_sims, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.success(f"Resultados salvos em {output_path}:\n{results.to_string(index=False)}")


if __name__ == "__main__":
    app()
