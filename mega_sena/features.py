"""Features derivadas dos sorteios: frequências, paridade e quadrantes do volante.

O volante da Mega-Sena tem 6 linhas por 10 colunas (1-10 na primeira linha, 51-60 na última).
Ele é dividido em quatro quadrantes de 15 dezenas cada, cortando entre a 3ª e a 4ª linhas e
entre a 5ª e a 6ª colunas::

         col 1-5      col 6-10
        +-----------+-----------+
    1-3 |    Q1     |    Q2     |   Q1: 1-5, 11-15, 21-25     Q2: 6-10, 16-20, 26-30
        +-----------+-----------+
    4-6 |    Q3     |    Q4     |   Q3: 31-35, 41-45, 51-55   Q4: 36-40, 46-50, 56-60
        +-----------+-----------+

Uso::

    python -m mega_sena.features   # grava data/processed/features.csv
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mega_sena.config import (
    DRAW_COLUMNS,
    FEATURES_FILE,
    PROCESSED_RESULTS_FILE,
    TOTAL_NUMBERS,
)
from mega_sena.dataset import load_processed

app = typer.Typer()

VOLANTE_ROWS, VOLANTE_COLS = 6, 10
N_QUADRANTS = 4
QUADRANT_COLUMNS = [f"q{i}" for i in range(1, N_QUADRANTS + 1)]


def draws_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extrai as dezenas sorteadas como matriz inteira de formato (concursos, 6)."""
    return df[DRAW_COLUMNS].to_numpy(dtype=int)


def quadrant_of(numbers: np.ndarray | int) -> np.ndarray:
    """Retorna o quadrante (1 a 4) de cada dezena, segundo o layout do volante."""
    numbers = np.asarray(numbers)
    row, col = np.divmod(numbers - 1, VOLANTE_COLS)
    return 1 + 2 * (row >= VOLANTE_ROWS // 2) + (col >= VOLANTE_COLS // 2)


def number_frequencies(draws: np.ndarray) -> pd.Series:
    """Conta quantas vezes cada dezena (1 a 60) foi sorteada.

    Returns:
        Série indexada pela dezena, com todas as 60 dezenas (inclusive as nunca sorteadas).
    """
    counts = np.bincount(draws.ravel(), minlength=TOTAL_NUMBERS + 1)[1:]
    return pd.Series(counts, index=pd.RangeIndex(1, TOTAL_NUMBERS + 1, name="dezena"), name="freq")


def cumulative_frequencies(draws: np.ndarray) -> pd.DataFrame:
    """Frequência acumulada de cada dezena ao longo dos concursos.

    Returns:
        DataFrame (concursos x 60): a linha ``t`` traz quantas vezes cada dezena saiu entre o
        primeiro concurso e o concurso ``t``, inclusive.
    """
    indicator = np.zeros((len(draws), TOTAL_NUMBERS + 1), dtype=int)
    np.put_along_axis(indicator, draws, 1, axis=1)
    return pd.DataFrame(indicator[:, 1:].cumsum(axis=0), columns=range(1, TOTAL_NUMBERS + 1))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as features de cada concurso.

    Args:
        df: Resultados processados (ver :func:`mega_sena.dataset.build_dataframe`).

    Returns:
        DataFrame com ``concurso``, ``data``, ``pares``, ``impares``, ``soma`` (soma das
        dezenas), ``amplitude`` (maior - menor dezena) e ``q1``..``q4`` (quantas dezenas
        caíram em cada quadrante).
    """
    draws = draws_matrix(df)
    evens = (draws % 2 == 0).sum(axis=1)
    quadrants = quadrant_of(draws)

    features = df[["concurso", "data"]].copy()
    features["pares"] = evens
    features["impares"] = draws.shape[1] - evens
    features["soma"] = draws.sum(axis=1)
    features["amplitude"] = draws.max(axis=1) - draws.min(axis=1)
    for q, column in enumerate(QUADRANT_COLUMNS, start=1):
        features[column] = (quadrants == q).sum(axis=1)
    return features


@app.command()
def main(
    input_path: Path = PROCESSED_RESULTS_FILE,
    output_path: Path = FEATURES_FILE,
):
    """Gera o arquivo de features a partir dos resultados processados."""
    logger.info("Gerando features...")
    features = build_features(load_processed(input_path))
    features.to_csv(output_path, index=False)
    logger.success(f"Features de {len(features)} concursos salvas em {output_path}.")


if __name__ == "__main__":
    app()
