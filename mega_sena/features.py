"""Features derivadas dos sorteios: frequências, paridade e quadrantes do volante.

O volante é lido como uma grade ``linhas x colunas`` (ver :attr:`GameSpec.volante`) e dividido
em quatro quadrantes, cortando na metade das linhas e na metade das colunas. Na Mega-Sena, com
6 linhas por 10 colunas, cada quadrante tem 15 dezenas::

         col 1-5      col 6-10
        +-----------+-----------+
    1-3 |    Q1     |    Q2     |   Q1: 1-5, 11-15, 21-25     Q2: 6-10, 16-20, 26-30
        +-----------+-----------+
    4-6 |    Q3     |    Q4     |   Q3: 31-35, 41-45, 51-55   Q4: 36-40, 46-50, 56-60
        +-----------+-----------+

Com número ímpar de linhas ou colunas, os quadrantes têm tamanhos diferentes; os testes usam o
tamanho real de cada um (:func:`quadrant_sizes`).

Uso::

    python -m mega_sena.features   # grava data/processed/<jogo>_features.csv
"""

import re

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mega_sena.dataset import load_processed
from mega_sena.games import MEGA_SENA, GameSpec, get_game

app = typer.Typer()

N_QUADRANTS = 4
QUADRANT_COLUMNS = [f"q{i}" for i in range(1, N_QUADRANTS + 1)]


def draws_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extrai as dezenas sorteadas (colunas ``d1``..``dK``) como matriz inteira (concursos, K)."""
    columns = [col for col in df.columns if re.fullmatch(r"d\d+", col)]
    return df[columns].to_numpy(dtype=int)


def quadrant_of(numbers: np.ndarray | int, spec: GameSpec = MEGA_SENA) -> np.ndarray:
    """Retorna o quadrante (1 a 4) de cada dezena, segundo o layout do volante."""
    rows, cols = spec.volante
    row, col = np.divmod(np.asarray(numbers) - spec.menor, cols)
    return 1 + 2 * (row >= rows // 2) + (col >= cols // 2)


def quadrant_sizes(spec: GameSpec = MEGA_SENA) -> np.ndarray:
    """Quantas dezenas do volante pertencem a cada quadrante."""
    return np.bincount(quadrant_of(spec.numbers, spec), minlength=N_QUADRANTS + 1)[1:]


def even_count(spec: GameSpec = MEGA_SENA) -> int:
    """Quantas dezenas pares existem no volante."""
    return int((spec.numbers % 2 == 0).sum())


def number_frequencies(draws: np.ndarray, spec: GameSpec = MEGA_SENA) -> pd.Series:
    """Conta quantas vezes cada dezena do volante foi sorteada.

    Returns:
        Série indexada pela dezena, com todas as dezenas (inclusive as nunca sorteadas).
    """
    counts = np.bincount(draws.ravel() - spec.menor, minlength=spec.n)
    return pd.Series(counts, index=pd.Index(spec.numbers, name="dezena"), name="freq")


def cumulative_frequencies(draws: np.ndarray, spec: GameSpec = MEGA_SENA) -> pd.DataFrame:
    """Frequência acumulada de cada dezena ao longo dos concursos.

    Returns:
        DataFrame (concursos x dezenas): a linha ``t`` traz quantas vezes cada dezena saiu entre
        o primeiro concurso e o concurso ``t``, inclusive.
    """
    indicator = np.zeros((len(draws), spec.n), dtype=int)
    np.put_along_axis(indicator, draws - spec.menor, 1, axis=1)
    return pd.DataFrame(indicator.cumsum(axis=0), columns=spec.numbers)


def build_features(df: pd.DataFrame, spec: GameSpec = MEGA_SENA) -> pd.DataFrame:
    """Calcula as features de cada concurso.

    Args:
        df: Resultados processados (ver :func:`mega_sena.dataset.build_dataframe`).
        spec: Jogo dos resultados.

    Returns:
        DataFrame com ``concurso``, ``data``, ``pares``, ``impares``, ``soma`` (soma das
        dezenas), ``amplitude`` (maior - menor dezena) e ``q1``..``q4`` (quantas dezenas
        caíram em cada quadrante).
    """
    draws = draws_matrix(df)
    evens = (draws % 2 == 0).sum(axis=1)
    quadrants = quadrant_of(draws, spec)

    features = df[["concurso", "data"]].copy()
    features["pares"] = evens
    features["impares"] = draws.shape[1] - evens
    features["soma"] = draws.sum(axis=1)
    features["amplitude"] = draws.max(axis=1) - draws.min(axis=1)
    for q, column in enumerate(QUADRANT_COLUMNS, start=1):
        features[column] = (quadrants == q).sum(axis=1)
    return features


def run(spec: GameSpec) -> pd.DataFrame:
    """Gera e grava o arquivo de features de um jogo."""
    logger.info(f"{spec.nome}: gerando features...")
    features = build_features(load_processed(spec), spec)
    features.to_csv(spec.paths.features, index=False)
    logger.success(f"Features de {len(features)} concursos salvas em {spec.paths.features}.")
    return features


@app.command()
def main(jogo: str = typer.Option(MEGA_SENA.slug, help="Jogo (slug da API).")):
    """Gera o arquivo de features a partir dos resultados processados."""
    run(get_game(jogo))


if __name__ == "__main__":
    app()
