"""Testes de hipótese: os sorteios são compatíveis com um sorteio uniforme e independente?

Hipótese nula (H0): em cada concurso, as 6 dezenas são uma amostra *sem reposição*, uniforme,
das 60 dezenas, independente dos concursos anteriores. Sob H0 vale que:

* cada dezena aparece num concurso com probabilidade 6/60 = 0,1;
* o número de dezenas de um grupo com K elementos (pares, um quadrante...) num concurso segue
  uma distribuição hipergeométrica (população 60, K "sucessos", 6 retiradas).

**Correção para amostragem sem reposição.** O teste qui-quadrado clássico (``scipy.stats.
chisquare``) supõe contagens multinomiais, isto é, sorteio *com* reposição. Na Mega-Sena as 6
dezenas de um concurso são distintas, e isso reduz a variância das contagens pelo fator de
correção de população finita ``(60 - 6) / (60 - 1) = 54/59``. Sem a correção, a estatística fica
menor do que deveria e o teste perde poder. Por isso dividimos a estatística por esse fator; a
covariância da hipergeométrica multivariada é exatamente a multinomial vezes o fator, então a
estatística corrigida volta a ser aproximadamente qui-quadrado. Como verificação independente da
aproximação, o teste de frequências também é feito por simulação de Monte Carlo.

Uso::

    python -m mega_sena.hypothesis   # grava reports/hypothesis_tests.csv
"""

from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats
import typer

from mega_sena.config import (
    HYPOTHESIS_TESTS_FILE,
    NUMBERS_PER_DRAW,
    PROCESSED_RESULTS_FILE,
    RANDOM_SEED,
    TOTAL_NUMBERS,
)
from mega_sena.dataset import load_processed
from mega_sena.features import N_QUADRANTS, build_features, draws_matrix, quadrant_of

app = typer.Typer()

FPC = (TOTAL_NUMBERS - NUMBERS_PER_DRAW) / (TOTAL_NUMBERS - 1)  # 54/59


@dataclass(frozen=True)
class TestResult:
    """Resultado de um teste de hipótese."""

    teste: str
    estatistica: float
    graus_liberdade: int
    p_valor: float
    observacao: str = ""


def random_draws(n_contests: int, rng: np.random.Generator) -> np.ndarray:
    """Simula concursos sob H0: 6 dezenas distintas e uniformes por concurso.

    Usa ``argpartition`` sobre ruído uniforme, que equivale a sortear sem reposição e é
    vetorizado (muito mais rápido que ``rng.choice`` em laço).

    Returns:
        Matriz (n_contests, 6) com dezenas de 1 a 60.
    """
    noise = rng.random((n_contests, TOTAL_NUMBERS))
    return np.argpartition(noise, NUMBERS_PER_DRAW, axis=1)[:, :NUMBERS_PER_DRAW] + 1


def hypergeom_pmf(successes: int) -> np.ndarray:
    """Probabilidade de ``k`` = 0..6 dezenas de um grupo de ``successes`` dezenas num concurso."""
    return stats.hypergeom(M=TOTAL_NUMBERS, n=successes, N=NUMBERS_PER_DRAW).pmf(
        np.arange(NUMBERS_PER_DRAW + 1)
    )


def _merge_sparse_bins(
    observed: np.ndarray, expected: np.ndarray, min_expected: float
) -> tuple[np.ndarray, np.ndarray]:
    """Agrupa categorias vizinhas até cada uma ter frequência esperada >= ``min_expected``.

    É a regra prática usual para a aproximação qui-quadrado ser válida. Uma sobra no final é
    somada à última categoria formada.
    """
    obs_bins, exp_bins = [], []
    acc_obs = acc_exp = 0.0
    for obs, exp in zip(observed, expected):
        acc_obs, acc_exp = acc_obs + obs, acc_exp + exp
        if acc_exp >= min_expected:
            obs_bins.append(acc_obs)
            exp_bins.append(acc_exp)
            acc_obs = acc_exp = 0.0
    if acc_exp > 0:
        obs_bins[-1] += acc_obs
        exp_bins[-1] += acc_exp
    return np.array(obs_bins), np.array(exp_bins)


def category_totals_test(
    observed: np.ndarray, group_sizes: np.ndarray, n_contests: int, name: str
) -> TestResult:
    """Qui-quadrado de aderência para o total de dezenas sorteadas em cada categoria.

    Args:
        observed: Total de dezenas sorteadas em cada categoria, somando todos os concursos.
        group_sizes: Quantas dezenas do volante pertencem a cada categoria (soma = 60).
        n_contests: Número de concursos.
        name: Nome do teste, para o relatório.

    Returns:
        Resultado com a estatística corrigida para amostragem sem reposição.
    """
    expected = n_contests * NUMBERS_PER_DRAW * np.asarray(group_sizes) / TOTAL_NUMBERS
    statistic = ((observed - expected) ** 2 / expected).sum() / FPC
    dof = len(observed) - 1
    return TestResult(name, statistic, dof, stats.chi2.sf(statistic, dof), "corrigido 54/59")


def distribution_test(
    observed_k: np.ndarray, successes: int, name: str, min_expected: float = 5.0
) -> TestResult:
    """Qui-quadrado de aderência da contagem por concurso à distribuição hipergeométrica.

    Args:
        observed_k: ``observed_k[k]`` = número de concursos com exatamente ``k`` dezenas do grupo.
        successes: Tamanho do grupo no volante (ex.: 30 pares, 15 dezenas por quadrante).
        name: Nome do teste, para o relatório.
        min_expected: Frequência esperada mínima por categoria; as raras são agrupadas.
    """
    observed_k = np.asarray(observed_k, dtype=float)
    expected_k = observed_k.sum() * hypergeom_pmf(successes)
    obs, exp = _merge_sparse_bins(observed_k, expected_k, min_expected)
    statistic, p_value = stats.chisquare(obs, exp)
    return TestResult(name, statistic, len(obs) - 1, p_value, f"{len(obs)} categorias")


def frequency_statistic(draws: np.ndarray) -> float:
    """Estatística qui-quadrado (sem correção) da frequência das 60 dezenas."""
    counts = np.bincount(draws.ravel(), minlength=TOTAL_NUMBERS + 1)[1:]
    expected = draws.size / TOTAL_NUMBERS
    return float(((counts - expected) ** 2 / expected).sum())


def monte_carlo_frequency_test(
    draws: np.ndarray, n_sims: int = 2000, seed: int = RANDOM_SEED
) -> TestResult:
    """Teste de uniformidade das frequências por simulação de Monte Carlo.

    Simula ``n_sims`` históricos com o mesmo número de concursos sob H0 e compara a estatística
    observada com a distribuição simulada. Não depende de aproximação assintótica nem da
    correção 54/59, por isso serve de verificação do teste analítico.
    """
    rng = np.random.default_rng(seed)
    observed = frequency_statistic(draws)
    simulated = np.array(
        [frequency_statistic(random_draws(len(draws), rng)) for _ in range(n_sims)]
    )
    p_value = (1 + (simulated >= observed).sum()) / (n_sims + 1)
    return TestResult(
        "Frequência das 60 dezenas (Monte Carlo)",
        observed,
        TOTAL_NUMBERS - 1,
        p_value,
        f"estatística sem correção; {n_sims} simulações",
    )


#: Janelas da análise por período. Foram escolhidas DEPOIS de ver o teste global rejeitar H0,
#: por isso os p-valores por período são exploratórios (pós-hoc), não confirmatórios.
PERIODS = [(1996, 2005), (2006, 2015), (2016, 2026)]


def frequency_tests_by_period(
    df: pd.DataFrame, periods: list[tuple[int, int]] = PERIODS
) -> list[TestResult]:
    """Teste de frequência das 60 dezenas restrito a cada intervalo de anos (inclusivo)."""
    results = []
    for first, last in periods:
        subset = draws_matrix(df[df["data"].dt.year.between(first, last)])
        counts = np.bincount(subset.ravel(), minlength=TOTAL_NUMBERS + 1)[1:]
        result = category_totals_test(
            counts,
            np.ones(TOTAL_NUMBERS),
            len(subset),
            f"Frequência das 60 dezenas, {first}-{last}",
        )
        results.append(
            TestResult(**{**asdict(result), "observacao": f"pós-hoc; {len(subset)} concursos"})
        )
    return results


def run_all_tests(df: pd.DataFrame, n_sims: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Executa a bateria de testes de hipótese sobre o histórico.

    Os quatro primeiros testes são os confirmatórios, definidos antes de olhar os dados; as
    linhas por período são exploratórias (ver :data:`PERIODS`).

    Returns:
        Uma linha por teste. P-valores altos significam que não há evidência contra H0.
    """
    draws = draws_matrix(df)
    features = build_features(df)
    n_contests = len(draws)
    counts = np.bincount(draws.ravel(), minlength=TOTAL_NUMBERS + 1)[1:]
    quadrant_totals = np.bincount(quadrant_of(draws).ravel(), minlength=N_QUADRANTS + 1)[1:]
    evens_k = np.bincount(features["pares"], minlength=NUMBERS_PER_DRAW + 1)

    results = [
        category_totals_test(
            counts, np.ones(TOTAL_NUMBERS), n_contests, "Frequência das 60 dezenas"
        ),
        monte_carlo_frequency_test(draws, n_sims, seed),
        distribution_test(evens_k, TOTAL_NUMBERS // 2, "Pares por concurso vs. hipergeométrica"),
        category_totals_test(
            quadrant_totals,
            np.full(N_QUADRANTS, TOTAL_NUMBERS // N_QUADRANTS),
            n_contests,
            "Total de dezenas por quadrante",
        ),
        *frequency_tests_by_period(df),
    ]
    return pd.DataFrame([asdict(result) for result in results])


@app.command()
def main(
    input_path: Path = PROCESSED_RESULTS_FILE,
    output_path: Path = HYPOTHESIS_TESTS_FILE,
    n_sims: int = typer.Option(2000, help="Simulações do teste de Monte Carlo."),
    seed: int = RANDOM_SEED,
):
    """Roda os testes de hipótese e salva a tabela de resultados."""
    logger.info("Rodando testes de hipótese...")
    results = run_all_tests(load_processed(input_path), n_sims, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.success(f"Resultados salvos em {output_path}:\n{results.to_string(index=False)}")


if __name__ == "__main__":
    app()
