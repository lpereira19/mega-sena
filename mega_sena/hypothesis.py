"""Testes de hipótese: os sorteios são compatíveis com um sorteio uniforme e independente?

Hipótese nula (H0): em cada concurso, as K dezenas são uma amostra *sem reposição*, uniforme,
das N dezenas do volante, independente dos concursos anteriores. Sob H0 vale que:

* cada dezena aparece num concurso com probabilidade K/N (na Mega-Sena, 6/60 = 0,1);
* o número de dezenas de um grupo com G elementos (pares, um quadrante...) num concurso segue
  uma distribuição hipergeométrica (população N, G "sucessos", K retiradas).

**Correção para amostragem sem reposição.** O teste qui-quadrado clássico (``scipy.stats.
chisquare``) supõe contagens multinomiais, isto é, sorteio *com* reposição. Como as K dezenas de
um concurso são distintas, a variância das contagens é menor pelo fator de correção de população
finita ``(N - K) / (N - 1)`` (na Mega-Sena, ``54/59``). Sem a correção, a estatística fica menor
do que deveria e o teste perde poder. Por isso dividimos a estatística por esse fator; a
covariância da hipergeométrica multivariada é exatamente a multinomial vezes o fator, então a
estatística corrigida volta a ser aproximadamente qui-quadrado. Como verificação independente da
aproximação, o teste de frequências também é feito por simulação de Monte Carlo.

Uso::

    python -m mega_sena.hypothesis   # grava reports/<jogo>/hypothesis_tests.csv
"""

from dataclasses import asdict, dataclass

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats
import typer

from mega_sena.config import RANDOM_SEED
from mega_sena.dataset import load_processed
from mega_sena.features import (
    build_features,
    draws_matrix,
    even_count,
    quadrant_of,
    quadrant_sizes,
)
from mega_sena.games import MEGA_SENA, GameSpec, get_game

app = typer.Typer()

# Nomes dos testes na tabela de resultados (também usados por mega_sena.report). ``{n}`` é o
# total de dezenas do volante.
FREQUENCY_TEST = "Frequência das {n} dezenas"
FREQUENCY_MC_TEST = "Frequência das {n} dezenas (Monte Carlo)"
EVEN_ODD_TEST = "Pares por concurso vs. hipergeométrica"
QUADRANT_TEST = "Total de dezenas por quadrante"


@dataclass(frozen=True)
class TestResult:
    """Resultado de um teste de hipótese."""

    teste: str
    estatistica: float
    graus_liberdade: int
    p_valor: float
    observacao: str = ""


def format_test_name(template: str, spec: GameSpec = MEGA_SENA) -> str:
    """Preenche o total de dezenas do volante no nome de um teste."""
    return template.format(n=spec.n)


def fpc(spec: GameSpec = MEGA_SENA) -> float:
    """Fator de correção de população finita, ``(N - K) / (N - 1)``."""
    return (spec.n - spec.sorteadas) / (spec.n - 1)


def random_draws(
    n_contests: int,
    rng: np.random.Generator,
    spec: GameSpec = MEGA_SENA,
    size: int | None = None,
) -> np.ndarray:
    """Simula concursos (ou apostas) sob H0: ``size`` dezenas distintas e uniformes por linha.

    Usa ``argpartition`` sobre ruído uniforme, que equivale a sortear sem reposição e é
    vetorizado (muito mais rápido que ``rng.choice`` em laço).

    Args:
        n_contests: Número de linhas.
        rng: Gerador aleatório.
        spec: Jogo (define o volante).
        size: Dezenas por linha; o padrão é o número de dezenas sorteadas do jogo.

    Returns:
        Matriz (n_contests, size) com dezenas do volante.
    """
    size = size or spec.sorteadas
    noise = rng.random((n_contests, spec.n))
    return np.argpartition(noise, size, axis=1)[:, :size] + spec.menor


def hypergeom_pmf(successes: int, spec: GameSpec = MEGA_SENA) -> np.ndarray:
    """Probabilidade de ``k`` = 0..K dezenas de um grupo de ``successes`` dezenas num concurso."""
    return stats.hypergeom(M=spec.n, n=successes, N=spec.sorteadas).pmf(
        np.arange(spec.sorteadas + 1)
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
    observed: np.ndarray,
    group_sizes: np.ndarray,
    n_contests: int,
    name: str,
    spec: GameSpec = MEGA_SENA,
) -> TestResult:
    """Qui-quadrado de aderência para o total de dezenas sorteadas em cada categoria.

    Args:
        observed: Total de dezenas sorteadas em cada categoria, somando todos os concursos.
        group_sizes: Quantas dezenas do volante pertencem a cada categoria (soma = N).
        n_contests: Número de concursos.
        name: Nome do teste, para o relatório.
        spec: Jogo dos concursos.

    Returns:
        Resultado com a estatística corrigida para amostragem sem reposição.
    """
    expected = n_contests * spec.sorteadas * np.asarray(group_sizes) / spec.n
    statistic = ((observed - expected) ** 2 / expected).sum() / fpc(spec)
    dof = len(observed) - 1
    note = f"corrigido {spec.n - spec.sorteadas}/{spec.n - 1}"
    return TestResult(name, statistic, dof, stats.chi2.sf(statistic, dof), note)


def pmf_test(
    observed: np.ndarray, pmf: np.ndarray, name: str, min_expected: float = 5.0
) -> TestResult:
    """Qui-quadrado de aderência de contagens observadas a uma distribuição teórica.

    Args:
        observed: ``observed[i]`` = número de concursos na categoria ``i``.
        pmf: Probabilidade teórica de cada categoria (mesmo tamanho de ``observed``, soma 1).
        name: Nome do teste, para o relatório.
        min_expected: Frequência esperada mínima por categoria; as raras são agrupadas.
    """
    observed = np.asarray(observed, dtype=float)
    expected = observed.sum() * np.asarray(pmf)
    obs, exp = _merge_sparse_bins(observed, expected, min_expected)
    statistic, p_value = stats.chisquare(obs, exp)
    return TestResult(name, statistic, len(obs) - 1, p_value, f"{len(obs)} categorias")


def distribution_test(
    observed_k: np.ndarray,
    successes: int,
    name: str,
    min_expected: float = 5.0,
    spec: GameSpec = MEGA_SENA,
) -> TestResult:
    """Qui-quadrado de aderência da contagem por concurso à distribuição hipergeométrica.

    Args:
        observed_k: ``observed_k[k]`` = número de concursos com exatamente ``k`` dezenas do grupo.
        successes: Tamanho do grupo no volante (ex.: 30 pares, 15 dezenas por quadrante).
        name: Nome do teste, para o relatório.
        min_expected: Frequência esperada mínima por categoria; as raras são agrupadas.
        spec: Jogo dos concursos.
    """
    return pmf_test(observed_k, hypergeom_pmf(successes, spec), name, min_expected)


def frequency_statistic(draws: np.ndarray, spec: GameSpec = MEGA_SENA) -> float:
    """Estatística qui-quadrado (sem correção) da frequência das dezenas do volante."""
    counts = np.bincount(draws.ravel() - spec.menor, minlength=spec.n)
    expected = draws.size / spec.n
    return float(((counts - expected) ** 2 / expected).sum())


def monte_carlo_frequency_test(
    draws: np.ndarray, n_sims: int = 2000, seed: int = RANDOM_SEED, spec: GameSpec = MEGA_SENA
) -> TestResult:
    """Teste de uniformidade das frequências por simulação de Monte Carlo.

    Simula ``n_sims`` históricos com o mesmo número de concursos sob H0 e compara a estatística
    observada com a distribuição simulada. Não depende de aproximação assintótica nem do fator
    de correção, por isso serve de verificação do teste analítico.
    """
    rng = np.random.default_rng(seed)
    observed = frequency_statistic(draws, spec)
    simulated = np.array(
        [frequency_statistic(random_draws(len(draws), rng, spec), spec) for _ in range(n_sims)]
    )
    p_value = (1 + (simulated >= observed).sum()) / (n_sims + 1)
    return TestResult(
        format_test_name(FREQUENCY_MC_TEST, spec),
        observed,
        spec.n - 1,
        p_value,
        f"estatística sem correção; {n_sims} simulações",
    )


def frequency_tests_by_period(
    df: pd.DataFrame,
    periods: tuple[tuple[int, int | None], ...] | None = None,
    spec: GameSpec = MEGA_SENA,
) -> list[TestResult]:
    """Teste de frequência das dezenas restrito a cada intervalo de anos (inclusivo).

    As janelas padrão vêm de :attr:`GameSpec.periodos`. Na Mega-Sena, foram escolhidas DEPOIS
    de ver o teste global rejeitar H0, por isso os p-valores por período são exploratórios
    (pós-hoc), não confirmatórios. ``None`` no fim de uma janela significa "até o ano mais
    recente dos dados".
    """
    results = []
    for first, last in spec.periodos if periods is None else periods:
        last = last or int(df["data"].dt.year.max())
        subset = draws_matrix(df[df["data"].dt.year.between(first, last)])
        counts = np.bincount(subset.ravel() - spec.menor, minlength=spec.n)
        result = category_totals_test(
            counts,
            np.ones(spec.n),
            len(subset),
            f"{format_test_name(FREQUENCY_TEST, spec)}, {first}-{last}",
            spec,
        )
        results.append(
            TestResult(**{**asdict(result), "observacao": f"pós-hoc; {len(subset)} concursos"})
        )
    return results


def run_all_tests(
    df: pd.DataFrame, n_sims: int = 2000, seed: int = RANDOM_SEED, spec: GameSpec = MEGA_SENA
) -> pd.DataFrame:
    """Executa a bateria de testes de hipótese sobre o histórico.

    Os quatro primeiros testes são os confirmatórios, definidos antes de olhar os dados; as
    linhas por período são exploratórias (ver :func:`frequency_tests_by_period`).

    Returns:
        Uma linha por teste. P-valores altos significam que não há evidência contra H0.
    """
    draws = draws_matrix(df)
    features = build_features(df, spec)
    n_contests = len(draws)
    counts = np.bincount(draws.ravel() - spec.menor, minlength=spec.n)
    sizes = quadrant_sizes(spec)
    quadrant_totals = np.bincount(quadrant_of(draws, spec).ravel(), minlength=len(sizes) + 1)[1:]
    evens_k = np.bincount(features["pares"], minlength=spec.sorteadas + 1)

    results = [
        category_totals_test(
            counts, np.ones(spec.n), n_contests, format_test_name(FREQUENCY_TEST, spec), spec
        ),
        monte_carlo_frequency_test(draws, n_sims, seed, spec),
        distribution_test(evens_k, even_count(spec), EVEN_ODD_TEST, spec=spec),
        category_totals_test(quadrant_totals, sizes, n_contests, QUADRANT_TEST, spec),
        *frequency_tests_by_period(df, spec=spec),
    ]
    return pd.DataFrame([asdict(result) for result in results])


def run(spec: GameSpec, n_sims: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Roda os testes de hipótese de um jogo e grava a tabela de resultados."""
    logger.info(f"{spec.nome}: rodando testes de hipótese...")
    results = run_all_tests(load_processed(spec), n_sims, seed, spec)
    output_path = spec.paths.hypothesis_tests
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.success(f"Resultados salvos em {output_path}:\n{results.to_string(index=False)}")
    return results


@app.command()
def main(
    jogo: str = typer.Option(MEGA_SENA.slug, help="Jogo (slug da API)."),
    n_sims: int = typer.Option(2000, help="Simulações do teste de Monte Carlo."),
    seed: int = RANDOM_SEED,
):
    """Roda os testes de hipótese e salva a tabela de resultados."""
    run(get_game(jogo), n_sims, seed)


if __name__ == "__main__":
    app()
