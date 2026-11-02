"""Backtest walk-forward: estratégias baseadas no histórico vs. apostas aleatórias.

Desenho do experimento:

1. Para cada concurso ``t`` a partir de ``start``, cada estratégia escolhe 6 dezenas usando
   **apenas** os concursos anteriores a ``t`` (sem vazamento de informação do futuro).
2. A aposta é comparada com o resultado real de ``t`` e registramos o número de acertos.
3. Sob a hipótese nula (sorteios independentes e uniformes), o número de acertos de *qualquer*
   aposta, seja qual for o critério de escolha, segue a hipergeométrica (60, 6, 6), com média
   exata de 0,6 acerto por concurso. Uma estratégia só "funciona" se rejeitar essa hipótese.
4. Duas formas de checar isso:

   * qui-quadrado de aderência da distribuição de acertos à hipergeométrica;
   * Monte Carlo: simulamos milhares de apostadores puramente aleatórios nos mesmos concursos e
     vemos onde a média de acertos da estratégia cai nessa distribuição nula (p-valor bicaudal).

Uso::

    python -m mega_sena.modeling.backtest
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from mega_sena.config import (
    BACKTEST_HITS_FILE,
    BACKTEST_NULL_FILE,
    BACKTEST_SUMMARY_FILE,
    NUMBERS_PER_DRAW,
    PROCESSED_RESULTS_FILE,
    RANDOM_SEED,
    TOTAL_NUMBERS,
)
from mega_sena.dataset import load_processed
from mega_sena.features import draws_matrix
from mega_sena.hypothesis import distribution_test, random_draws

app = typer.Typer()

#: Uma estratégia recebe o histórico anterior (concursos, 6) e um gerador aleatório e devolve as
#: 6 dezenas apostadas.
Strategy = Callable[[np.ndarray, np.random.Generator], np.ndarray]

EXPECTED_HITS = NUMBERS_PER_DRAW * NUMBERS_PER_DRAW / TOTAL_NUMBERS  # 0,6


def _ranked_by_frequency(
    past: np.ndarray, rng: np.random.Generator, window: int | None, hottest: bool
) -> np.ndarray:
    """Escolhe as 6 dezenas mais (ou menos) frequentes nos últimos ``window`` concursos.

    Empates são desfeitos ao acaso. Desempatar pela menor dezena enviesaria a aposta para os
    números baixos.
    """
    recent = past if window is None else past[-window:]
    counts = np.bincount(recent.ravel(), minlength=TOTAL_NUMBERS + 1)[1:]
    key = -counts if hottest else counts
    order = np.lexsort((rng.random(TOTAL_NUMBERS), key))  # última chave = critério principal
    return np.sort(order[:NUMBERS_PER_DRAW] + 1)


def hot_strategy(window: int | None = None) -> Strategy:
    """Aposta nas dezenas "quentes", as mais sorteadas (no histórico todo ou numa janela)."""
    return partial(_ranked_by_frequency, window=window, hottest=True)


def cold_strategy(window: int | None = None) -> Strategy:
    """Aposta nas dezenas "frias", as menos sorteadas ("estão atrasadas, vão sair")."""
    return partial(_ranked_by_frequency, window=window, hottest=False)


def random_strategy(past: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aposta puramente aleatória (gerador pseudoaleatório), que ignora o histórico."""
    return np.sort(rng.choice(TOTAL_NUMBERS, NUMBERS_PER_DRAW, replace=False) + 1)


DEFAULT_STRATEGIES: dict[str, Strategy] = {
    "quentes (histórico)": hot_strategy(),
    "quentes (últimos 50)": hot_strategy(window=50),
    "frias (histórico)": cold_strategy(),
    "aleatória": random_strategy,
}


def count_hits(bets: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Número de acertos de cada aposta contra o sorteio da mesma linha.

    Args:
        bets: Apostas (concursos, 6).
        draws: Resultados (concursos, 6).
    """
    return (bets[:, :, None] == draws[:, None, :]).any(axis=2).sum(axis=1)


def run_backtest(
    draws: np.ndarray, strategy: Strategy, start: int = 100, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Simula a estratégia em walk-forward e retorna os acertos por concurso.

    Args:
        draws: Histórico (concursos, 6) em ordem cronológica.
        strategy: Função que gera a aposta a partir do histórico anterior.
        start: Primeiro concurso apostado (índice 0-based); os anteriores formam o histórico
            inicial.
        seed: Semente do gerador aleatório, para reprodutibilidade.

    Returns:
        Vetor de acertos (0 a 6) para os concursos ``start`` .. fim.
    """
    rng = np.random.default_rng(seed)
    bets = np.array([strategy(draws[:t], rng) for t in range(start, len(draws))])
    return count_hits(bets, draws[start:])


def null_mean_hits(draws: np.ndarray, n_sims: int = 2000, seed: int = RANDOM_SEED) -> np.ndarray:
    """Distribuição nula da média de acertos: ``n_sims`` apostadores aleatórios.

    Args:
        draws: Concursos avaliados (os mesmos do backtest).
        n_sims: Número de apostadores aleatórios simulados.

    Returns:
        Vetor com a média de acertos de cada apostador simulado.
    """
    rng = np.random.default_rng(seed)
    return np.array(
        [count_hits(random_draws(len(draws), rng), draws).mean() for _ in range(n_sims)]
    )


def summarize(hits: np.ndarray, null_means: np.ndarray, name: str) -> dict:
    """Resume o desempenho de uma estratégia e testa se ela difere do acaso.

    Returns:
        Dicionário com média de acertos, contagem de quadras/quinas/senas e os p-valores do
        qui-quadrado (distribuição de acertos) e do Monte Carlo (média de acertos, bicaudal).
    """
    hits_k = np.bincount(hits, minlength=NUMBERS_PER_DRAW + 1)
    gof = distribution_test(hits_k, NUMBERS_PER_DRAW, name)
    deviation = abs(hits.mean() - EXPECTED_HITS)
    p_mc = (1 + (np.abs(null_means - EXPECTED_HITS) >= deviation).sum()) / (len(null_means) + 1)
    return {
        "estrategia": name,
        "apostas": len(hits),
        "media_acertos": hits.mean(),
        "quadras": int(hits_k[4]),
        "quinas": int(hits_k[5]),
        "senas": int(hits_k[6]),
        "p_valor_qui2": gof.p_valor,
        "p_valor_monte_carlo": p_mc,
    }


def compare_strategies(
    df: pd.DataFrame,
    strategies: dict[str, Strategy] = DEFAULT_STRATEGIES,
    start: int = 100,
    n_sims: int = 2000,
    seed: int = RANDOM_SEED,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Roda o backtest de todas as estratégias nos mesmos concursos.

    Returns:
        ``(resumo, acertos, medias_nulas)``: uma linha por estratégia, com o p-valor do Monte
        Carlo também corrigido por Bonferroni; os acertos por concurso (formato longo); e a
        distribuição nula das médias de acertos.
    """
    draws = draws_matrix(df)
    null_means = null_mean_hits(draws[start:], n_sims, seed)

    summaries, hits_frames = [], []
    for name, strategy in tqdm(strategies.items(), desc="estratégias", disable=not progress):
        hits = run_backtest(draws, strategy, start, seed)
        summaries.append(summarize(hits, null_means, name))
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


@app.command()
def main(
    input_path: Path = PROCESSED_RESULTS_FILE,
    summary_path: Path = BACKTEST_SUMMARY_FILE,
    hits_path: Path = BACKTEST_HITS_FILE,
    null_path: Path = BACKTEST_NULL_FILE,
    start: int = typer.Option(100, help="Concursos usados só como histórico inicial."),
    n_sims: int = typer.Option(2000, help="Apostadores aleatórios na distribuição nula."),
    seed: int = RANDOM_SEED,
):
    """Executa o backtest das estratégias padrão e salva resumo, acertos e distribuição nula."""
    logger.info("Rodando backtest walk-forward...")
    summary, hits, null_means = compare_strategies(
        load_processed(input_path), start=start, n_sims=n_sims, seed=seed
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    hits.to_csv(hits_path, index=False)
    pd.Series(null_means, name="media_acertos").to_csv(null_path, index=False)
    logger.success(
        f"Backtest concluído (esperado: {EXPECTED_HITS:.2f} acerto/concurso):\n"
        f"{summary.to_string(index=False)}"
    )


if __name__ == "__main__":
    app()
