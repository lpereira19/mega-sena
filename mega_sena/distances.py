"""Distância entre as dezenas de um mesmo concurso.

Perguntas respondidas:

* Com que frequência saem dezenas **consecutivas** (ex.: 22 e 23) no mesmo concurso?
* Qual a **menor distância** entre duas dezenas de um concurso, e como se distribuem os saltos
  entre dezenas vizinhas na ordem crescente?
* **Quando sai a dezena x**, a que distância ficam as outras? A vizinha mais próxima fica mais
  perto ou mais longe do que o acaso prevê?

Sob H0 (K dezenas distintas e uniformes entre N), todas essas distribuições têm fórmula exata,
obtida por contagem de combinações; elas dependem só de N e K. As fórmulas estão nas funções
``*_pmf`` e são conferidas contra simulação nos testes automatizados.

Uso::

    python -m mega_sena.distances   # grava reports/<jogo>/distance_tests.csv
"""

from dataclasses import asdict
from functools import cache
from itertools import pairwise
from math import comb

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mega_sena.config import RANDOM_SEED
from mega_sena.dataset import load_processed
from mega_sena.features import draws_matrix
from mega_sena.games import MEGA_SENA, GameSpec, get_game
from mega_sena.hypothesis import TestResult, pmf_test, random_draws

app = typer.Typer()

# Nomes dos testes na tabela de resultados (também usados por mega_sena.report)
CONSECUTIVE_TEST = "Pares consecutivos por concurso"
LONGEST_RUN_TEST = "Maior sequência de consecutivas por concurso"
MIN_GAP_TEST = "Menor distância por concurso"
GAP_TEST = "Saltos entre dezenas vizinhas (Monte Carlo)"
PAIR_TEST = "Distância entre todos os pares (Monte Carlo)"
NEAREST_TEST = "Vizinha mais próxima de cada dezena (Monte Carlo)"


def _mc_dof(spec: GameSpec) -> dict[str, int]:
    """Graus de liberdade de referência dos testes de Monte Carlo (o p-valor não depende deles)."""
    return {GAP_TEST: spec.teto_salto - 1, PAIR_TEST: spec.n - 2, NEAREST_TEST: spec.n}


# ---------------------------------------------------------------------------------------------
# Métricas observadas
# ---------------------------------------------------------------------------------------------


def sorted_gaps(draws: np.ndarray) -> np.ndarray:
    """Saltos entre dezenas vizinhas na ordem crescente: matriz (concursos, K - 1).

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
    """Distância entre cada par de dezenas de cada concurso: matriz (concursos, K(K-1)/2)."""
    rows, cols = np.triu_indices(draws.shape[1], k=1)
    return np.abs(draws[:, rows] - draws[:, cols])


def nearest_neighbor(draws: np.ndarray) -> np.ndarray:
    """Para cada dezena sorteada, a distância até a outra dezena mais próxima do mesmo concurso.

    Returns:
        Matriz (concursos, K), alinhada com ``draws``.
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


def nearest_by_number(draws: np.ndarray, spec: GameSpec = MEGA_SENA) -> pd.DataFrame:
    """Distância média até a vizinha mais próxima, por dezena, comparada ao esperado.

    Returns:
        Uma linha por dezena: ``vezes`` (quantas vezes saiu), ``media`` observada,
        ``esperado`` e ``desvio_padrao`` teóricos de uma ocorrência, e ``z`` (desvio da média
        em erros-padrão).
    """
    positions = draws.ravel() - spec.menor
    counts = np.bincount(positions, minlength=spec.n)
    sums = np.bincount(positions, weights=nearest_neighbor(draws).ravel(), minlength=spec.n)
    table = nearest_theory(spec).copy()
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
def nearest_theory(spec: GameSpec = MEGA_SENA) -> pd.DataFrame:
    """Média e desvio-padrão teóricos da distância até a vizinha mais próxima, por dezena.

    As dezenas das pontas do volante têm vizinhas só de um lado, por isso a distância esperada
    é maior para elas.
    """
    return pd.DataFrame(
        [_mean_sd(nearest_neighbor_pmf(x, spec)) for x in spec.numbers],
        index=pd.Index(spec.numbers, name="dezena"),
        columns=["esperado", "desvio_padrao"],
    )


@cache
def consecutive_pairs_pmf(spec: GameSpec = MEGA_SENA) -> pd.Series:
    """P(o concurso ter exatamente j pares consecutivos), j = 0..K-1.

    Fórmula clássica de "sucessões": C(K-1, j) * C(N-K+1, K-j) / C(N, K). Na Mega-Sena, para
    j = 0 dá C(55, 6) / C(60, 6) ≈ 0,579, ou seja, cerca de 42% dos concursos têm ao menos um
    par de dezenas consecutivas.
    """
    n, k = spec.n, spec.sorteadas
    pmf = [comb(k - 1, j) * comb(n - k + 1, k - j) / comb(n, k) for j in range(k)]
    return pd.Series(pmf, index=pd.RangeIndex(k, name="pares_consecutivos"))


def _count_max_run_at_most(r: int, spec: GameSpec) -> int:
    """Quantas combinações de K dezenas entre N não têm sequência de consecutivas maior que r.

    Programação dinâmica percorrendo o volante do início ao fim: o estado é (dezenas já
    escolhidas, tamanho da sequência atual terminando na posição corrente).
    """
    k = spec.sorteadas
    ways = np.zeros((k + 1, r + 1), dtype=object)  # object: inteiros exatos, sem overflow
    ways[0, 0] = 1
    for _ in range(spec.n):
        new = np.zeros_like(ways)
        new[:, 0] = ways.sum(axis=1)  # posição não escolhida: a sequência zera
        new[1:, 1:] = ways[:-1, :-1]  # posição escolhida: a sequência cresce
        ways = new
    return int(ways[k].sum())


@cache
def longest_run_pmf(spec: GameSpec = MEGA_SENA) -> pd.Series:
    """P(maior sequência de dezenas consecutivas do concurso = r), r = 1..K (exata)."""
    k = spec.sorteadas
    at_most = [_count_max_run_at_most(r, spec) for r in range(k + 1)]
    total = comb(spec.n, k)
    # Divisão com inteiros exatos do Python: em volantes grandes (Lotomania), as contagens
    # passam de 2**63 e o numpy não as representaria.
    pmf = np.array([(b - a) / total for a, b in pairwise(at_most)])
    return pd.Series(pmf, index=pd.RangeIndex(1, k + 1, name="maior_sequencia"))


@cache
def min_gap_pmf(spec: GameSpec = MEGA_SENA) -> pd.Series:
    """P(menor distância do concurso = g).

    Subtraindo (g-1)·i da i-ésima menor dezena, os concursos com todos os saltos >= g viram
    combinações quaisquer de N - (g-1)(K-1) números: P(menor >= g) = C(N - (g-1)(K-1), K) / C(N, K).
    """
    n, k = spec.n, spec.sorteadas
    survival = []
    g = 1
    while (remaining := n - (g - 1) * (k - 1)) >= k:
        survival.append(comb(remaining, k) / comb(n, k))
        g += 1
    survival.append(0.0)
    pmf = -np.diff(survival)
    return pd.Series(pmf, index=pd.RangeIndex(1, len(pmf) + 1, name="menor_distancia"))


@cache
def gap_pmf(spec: GameSpec = MEGA_SENA) -> pd.Series:
    """P(salto entre duas dezenas vizinhas na ordem crescente = g), igual para todos os saltos.

    Juntar duas vizinhas a distância g num só bloco deixa C(N - g, K - 1) maneiras de escolher o
    resto, logo P(salto = g) = C(N - g, K - 1) / C(N, K). Saltos >= ``spec.teto_salto`` são
    somados na última categoria.
    """
    n, k, cap = spec.n, spec.sorteadas, spec.teto_salto
    pmf = np.array([comb(n - g, k - 1) / comb(n, k) for g in range(1, n - k + 2)])
    capped = np.append(pmf[: cap - 1], pmf[cap - 1 :].sum())
    return pd.Series(capped, index=pd.RangeIndex(1, cap + 1, name="salto"))


@cache
def pair_distance_pmf(spec: GameSpec = MEGA_SENA) -> pd.Series:
    """P(distância entre duas dezenas de um concurso = d) = (N - d) / C(N, 2), d = 1..N-1.

    Há N - d pares de dezenas a distância d no volante, e todos os C(N, 2) pares são igualmente
    prováveis. Distâncias curtas são mais comuns só porque existem mais pares assim.
    """
    n = spec.n
    d = np.arange(1, n)
    return pd.Series((n - d) / comb(n, 2), index=pd.RangeIndex(1, n, name="distancia"))


def nearest_neighbor_pmf(number: int, spec: GameSpec = MEGA_SENA) -> pd.Series:
    """P(a vizinha mais próxima da dezena ``number`` estar a distância d), dado que ela saiu.

    As outras K - 1 dezenas são uma amostra uniforme das N - 1 restantes. Se ``m(d)`` dezenas
    ficam a menos de d de ``number`` (menos nas pontas do volante), então
    P(vizinha >= d) = C(N - 1 - m(d), K - 1) / C(N - 1, K - 1).
    """
    n, k = spec.n, spec.sorteadas
    position = number - spec.menor + 1  # 1..N
    others = n - 1
    survival = []
    for d in range(1, n + 1):
        within = min(d - 1, position - 1) + min(d - 1, n - position)
        survival.append(comb(others - within, k - 1) / comb(others, k - 1))
    pmf = -np.diff(survival)
    return pd.Series(pmf, index=pd.RangeIndex(1, n, name="distancia"))


# ---------------------------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------------------------


def _chi2(counts: np.ndarray, expected: np.ndarray) -> float:
    return float(((counts - expected) ** 2 / expected).sum())


def _dependent_statistics(draws: np.ndarray, spec: GameSpec) -> dict[str, float]:
    """Estatísticas cujas observações não são independentes dentro de um concurso.

    Os saltos, os pares e as vizinhas de um mesmo concurso são correlacionados, então o
    qui-quadrado não segue a distribuição teórica; o p-valor vem de simulação (Monte Carlo).
    """
    n_contests, k = draws.shape
    cap = spec.teto_salto
    gaps = np.minimum(sorted_gaps(draws), cap).ravel()
    pairs = pair_distances(draws)
    nearest = nearest_by_number(draws, spec)
    return {
        GAP_TEST: _chi2(
            np.bincount(gaps, minlength=cap + 1)[1:],
            gap_pmf(spec).to_numpy() * n_contests * (k - 1),
        ),
        PAIR_TEST: _chi2(
            np.bincount(pairs.ravel(), minlength=spec.n)[1:],
            pair_distance_pmf(spec).to_numpy() * n_contests * pairs.shape[1],
        ),
        NEAREST_TEST: float((nearest["z"].dropna() ** 2).sum()),
    }


def run_distance_tests(
    df: pd.DataFrame, n_sims: int = 2000, seed: int = RANDOM_SEED, spec: GameSpec = MEGA_SENA
) -> pd.DataFrame:
    """Testa se as distâncias entre as dezenas seguem o esperado num sorteio uniforme.

    Os três primeiros testes usam uma observação por concurso (independentes) e o qui-quadrado
    tradicional. Os demais usam várias observações correlacionadas por concurso e p-valor de
    Monte Carlo. Como são vários testes, a coluna ``p_valor_bonferroni`` corrige para múltiplas
    comparações.

    Returns:
        Uma linha por teste.
    """
    draws = draws_matrix(df)
    k = spec.sorteadas
    per_contest = [
        (
            np.bincount(consecutive_pairs(draws), minlength=k),
            consecutive_pairs_pmf(spec),
            CONSECUTIVE_TEST,
        ),
        (
            np.bincount(longest_run(draws), minlength=k + 1)[1:],
            longest_run_pmf(spec),
            LONGEST_RUN_TEST,
        ),
        (
            np.bincount(min_gap(draws), minlength=len(min_gap_pmf(spec)) + 1)[1:],
            min_gap_pmf(spec),
            MIN_GAP_TEST,
        ),
    ]
    # Métricas sem variação possível não têm o que testar: na Lotofácil (15 de 25) sempre sai
    # pelo menos um par consecutivo, então a menor distância é sempre 1.
    results = [
        pmf_test(observed, pmf, name)
        for observed, pmf, name in per_contest
        if (np.asarray(pmf) > 0).sum() > 1
    ]

    rng = np.random.default_rng(seed)
    observed = _dependent_statistics(draws, spec)
    simulated = [
        _dependent_statistics(random_draws(len(draws), rng, spec), spec) for _ in range(n_sims)
    ]
    dof = _mc_dof(spec)
    for name, value in observed.items():
        null = np.array([sim[name] for sim in simulated])
        p_value = (1 + (null >= value).sum()) / (n_sims + 1)
        results.append(TestResult(name, value, dof[name], p_value, f"{n_sims} simulações"))

    table = pd.DataFrame([asdict(result) for result in results])
    table["p_valor_bonferroni"] = (table["p_valor"] * len(table)).clip(upper=1.0)
    return table


def run(spec: GameSpec, n_sims: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Roda os testes de distância de um jogo e grava a tabela de resultados."""
    logger.info(f"{spec.nome}: rodando testes de distância entre as dezenas...")
    results = run_distance_tests(load_processed(spec), n_sims, seed, spec)
    output_path = spec.paths.distance_tests
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.success(f"Resultados salvos em {output_path}:\n{results.to_string(index=False)}")
    return results


@app.command()
def main(
    jogo: str = typer.Option(MEGA_SENA.slug, help="Jogo (slug da API)."),
    n_sims: int = typer.Option(2000, help="Simulações dos testes de Monte Carlo."),
    seed: int = RANDOM_SEED,
):
    """Roda os testes de distância entre as dezenas e salva a tabela de resultados."""
    run(get_game(jogo), n_sims, seed)


if __name__ == "__main__":
    app()
