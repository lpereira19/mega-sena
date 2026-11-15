"""Backtest walk-forward: estratégias baseadas no histórico vs. apostas aleatórias.

Desenho do experimento:

1. Para cada concurso ``t`` a partir de ``start``, cada estratégia escolhe B dezenas (a aposta
   mínima do jogo) usando **apenas** os concursos anteriores a ``t`` (sem vazamento de
   informação do futuro).
2. A aposta é comparada com o resultado real de ``t`` e registramos o número de acertos.
3. Sob a hipótese nula (sorteios independentes e uniformes), o número de acertos de *qualquer*
   aposta, seja qual for o critério de escolha, segue a hipergeométrica (N, B, K), com média
   exata de B·K/N acerto por concurso (0,6 na Mega-Sena). Uma estratégia só "funciona" se
   rejeitar essa hipótese.
4. Duas formas de checar isso:

   * qui-quadrado de aderência da distribuição de acertos à hipergeométrica;
   * Monte Carlo: simulamos milhares de apostadores puramente aleatórios nos mesmos concursos e
     vemos onde a média de acertos da estratégia cai nessa distribuição nula (p-valor bicaudal).

Uso::

    python -m mega_sena.modeling.backtest
"""

from collections.abc import Callable
from functools import partial

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import typer

from mega_sena.config import RANDOM_SEED
from mega_sena.dataset import load_processed
from mega_sena.features import draws_matrix
from mega_sena.games import MEGA_SENA, GameSpec, get_game
from mega_sena.hypothesis import pmf_test, random_draws

app = typer.Typer()

#: Uma estratégia recebe o histórico anterior (concursos, K) e um gerador aleatório e devolve as
#: B dezenas apostadas.
Strategy = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def expected_hits(spec: GameSpec = MEGA_SENA) -> float:
    """Média exata de acertos por concurso de qualquer aposta sob H0: B·K/N."""
    return spec.aposta * spec.sorteadas / spec.n


def hits_pmf(spec: GameSpec = MEGA_SENA) -> np.ndarray:
    """P(k acertos), k = 0..K, de uma aposta de B dezenas: hipergeométrica (N, B, K)."""
    return stats.hypergeom(M=spec.n, n=spec.aposta, N=spec.sorteadas).pmf(
        np.arange(spec.sorteadas + 1)
    )


def _ranked_by_frequency(
    past: np.ndarray,
    rng: np.random.Generator,
    window: int | None,
    hottest: bool,
    spec: GameSpec,
) -> np.ndarray:
    """Escolhe as B dezenas mais (ou menos) frequentes nos últimos ``window`` concursos.

    Empates são desfeitos ao acaso. Desempatar pela menor dezena enviesaria a aposta para os
    números baixos.
    """
    recent = past if window is None else past[-window:]
    counts = np.bincount(recent.ravel() - spec.menor, minlength=spec.n)
    key = -counts if hottest else counts
    order = np.lexsort((rng.random(spec.n), key))  # última chave = critério principal
    return np.sort(order[: spec.aposta] + spec.menor)


def hot_strategy(window: int | None = None, spec: GameSpec = MEGA_SENA) -> Strategy:
    """Aposta nas dezenas "quentes", as mais sorteadas (no histórico todo ou numa janela)."""
    return partial(_ranked_by_frequency, window=window, hottest=True, spec=spec)


def cold_strategy(window: int | None = None, spec: GameSpec = MEGA_SENA) -> Strategy:
    """Aposta nas dezenas "frias", as menos sorteadas ("estão atrasadas, vão sair")."""
    return partial(_ranked_by_frequency, window=window, hottest=False, spec=spec)


def random_strategy(
    past: np.ndarray, rng: np.random.Generator, spec: GameSpec = MEGA_SENA
) -> np.ndarray:
    """Aposta puramente aleatória (gerador pseudoaleatório), que ignora o histórico."""
    return np.sort(rng.choice(spec.n, spec.aposta, replace=False) + spec.menor)


def default_strategies(spec: GameSpec = MEGA_SENA) -> dict[str, Strategy]:
    """Estratégias comparadas no backtest padrão."""
    return {
        "quentes (histórico)": hot_strategy(spec=spec),
        "quentes (últimos 50)": hot_strategy(window=50, spec=spec),
        "frias (histórico)": cold_strategy(spec=spec),
        "aleatória": partial(random_strategy, spec=spec),
    }


def count_hits(bets: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Número de acertos de cada aposta contra o sorteio da mesma linha.

    Args:
        bets: Apostas (concursos, B).
        draws: Resultados (concursos, K).
    """
    return (bets[:, :, None] == draws[:, None, :]).any(axis=2).sum(axis=1)


def run_backtest(
    draws: np.ndarray, strategy: Strategy, start: int = 100, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Simula a estratégia em walk-forward e retorna os acertos por concurso.

    Args:
        draws: Histórico (concursos, K) em ordem cronológica.
        strategy: Função que gera a aposta a partir do histórico anterior.
        start: Primeiro concurso apostado (índice 0-based); os anteriores formam o histórico
            inicial.
        seed: Semente do gerador aleatório, para reprodutibilidade.

    Returns:
        Vetor de acertos (0 a K) para os concursos ``start`` .. fim.
    """
    rng = np.random.default_rng(seed)
    bets = np.array([strategy(draws[:t], rng) for t in range(start, len(draws))])
    return count_hits(bets, draws[start:])


def null_mean_hits(
    draws: np.ndarray, n_sims: int = 2000, seed: int = RANDOM_SEED, spec: GameSpec = MEGA_SENA
) -> np.ndarray:
    """Distribuição nula da média de acertos: ``n_sims`` apostadores aleatórios.

    Args:
        draws: Concursos avaliados (os mesmos do backtest).
        n_sims: Número de apostadores aleatórios simulados.
        seed: Semente do gerador aleatório.
        spec: Jogo (define o volante e o tamanho da aposta).

    Returns:
        Vetor com a média de acertos de cada apostador simulado.
    """
    rng = np.random.default_rng(seed)
    return np.array(
        [
            count_hits(random_draws(len(draws), rng, spec, size=spec.aposta), draws).mean()
            for _ in range(n_sims)
        ]
    )


def summarize(
    hits: np.ndarray, null_means: np.ndarray, name: str, spec: GameSpec = MEGA_SENA
) -> dict:
    """Resume o desempenho de uma estratégia e testa se ela difere do acaso.

    Returns:
        Dicionário com média de acertos, contagem das faixas de prêmio do jogo (ver
        :attr:`GameSpec.faixas`) e os p-valores do qui-quadrado (distribuição de acertos) e do
        Monte Carlo (média de acertos, bicaudal).
    """
    hits_k = np.bincount(hits, minlength=spec.sorteadas + 1)
    gof = pmf_test(hits_k, hits_pmf(spec), name)
    expected = expected_hits(spec)
    deviation = abs(hits.mean() - expected)
    p_mc = (1 + (np.abs(null_means - expected) >= deviation).sum()) / (len(null_means) + 1)
    return {
        "estrategia": name,
        "apostas": len(hits),
        "media_acertos": hits.mean(),
        **{column: int(hits_k[k]) for column, k in spec.faixas},
        "p_valor_qui2": gof.p_valor,
        "p_valor_monte_carlo": p_mc,
    }


def compare_strategies(
    df: pd.DataFrame,
    strategies: dict[str, Strategy] | None = None,
    start: int = 100,
    n_sims: int = 2000,
    seed: int = RANDOM_SEED,
    progress: bool = True,
    spec: GameSpec = MEGA_SENA,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Roda o backtest de todas as estratégias nos mesmos concursos.

    Args:
        df: Resultados processados do jogo.
        strategies: Estratégias a comparar; o padrão é :func:`default_strategies`.
        start: Concursos usados só como histórico inicial.
        n_sims: Apostadores aleatórios na distribuição nula.
        seed: Semente do gerador aleatório.
        progress: Mostra barra de progresso.
        spec: Jogo dos resultados.

    Returns:
        ``(resumo, acertos, medias_nulas)``: uma linha por estratégia, com o p-valor do Monte
        Carlo também corrigido por Bonferroni; os acertos por concurso (formato longo); e a
        distribuição nula das médias de acertos.
    """
    strategies = strategies or default_strategies(spec)
    draws = draws_matrix(df)
    null_means = null_mean_hits(draws[start:], n_sims, seed, spec)

    summaries, hits_frames = [], []
    for name, strategy in tqdm(strategies.items(), desc="estratégias", disable=not progress):
        hits = run_backtest(draws, strategy, start, seed)
        summaries.append(summarize(hits, null_means, name, spec))
        hits_frames.append(
            pd.DataFrame(
                {"concurso": df["concurso"].iloc[start:], "estrategia": name, "acertos": hits}
            )
        )
    summary = pd.DataFrame(summaries)
    # Testamos várias estratégias ao mesmo tempo: sem correção, a chance de alguma "funcionar"
    # por acaso cresce com o número de estratégias. Bonferroni é conservador, mas simples.
    summary["p_valor_mc_bonferroni"] = (summary["p_valor_monte_carlo"] * len(summary)).clip(
        upper=1.0
    )
    return summary, pd.concat(hits_frames, ignore_index=True), null_means


def run(
    spec: GameSpec, start: int = 100, n_sims: int = 2000, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Executa o backtest padrão de um jogo e grava resumo, acertos e distribuição nula."""
    logger.info(f"{spec.nome}: rodando backtest walk-forward...")
    summary, hits, null_means = compare_strategies(
        load_processed(spec), start=start, n_sims=n_sims, seed=seed, spec=spec
    )
    paths = spec.paths
    paths.backtest_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(paths.backtest_summary, index=False)
    hits.to_csv(paths.backtest_hits, index=False)
    pd.Series(null_means, name="media_acertos").to_csv(paths.backtest_null, index=False)
    logger.success(
        f"Backtest concluído (esperado: {expected_hits(spec):.2f} acerto/concurso):\n"
        f"{summary.to_string(index=False)}"
    )
    return summary


@app.command()
def main(
    jogo: str = typer.Option(MEGA_SENA.slug, help="Jogo (slug da API)."),
    start: int = typer.Option(100, help="Concursos usados só como histórico inicial."),
    n_sims: int = typer.Option(2000, help="Apostadores aleatórios na distribuição nula."),
    seed: int = RANDOM_SEED,
):
    """Executa o backtest das estratégias padrão e salva resumo, acertos e distribuição nula."""
    run(get_game(jogo), start, n_sims, seed)


if __name__ == "__main__":
    app()
