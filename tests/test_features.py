import numpy as np
import pandas as pd

from mega_sena.features import build_features, cumulative_frequencies, quadrant_of


def test_quadrant_layout():
    assert quadrant_of(np.array([1, 5, 25])).tolist() == [1, 1, 1]
    assert quadrant_of(np.array([6, 10, 30])).tolist() == [2, 2, 2]
    assert quadrant_of(np.array([31, 35, 55])).tolist() == [3, 3, 3]
    assert quadrant_of(np.array([36, 40, 60])).tolist() == [4, 4, 4]


def test_each_quadrant_has_15_numbers():
    assert np.bincount(quadrant_of(np.arange(1, 61)))[1:].tolist() == [15, 15, 15, 15]


def test_build_features():
    df = pd.DataFrame(
        {"concurso": [1], "data": [pd.Timestamp("2000-01-01")],
         "d1": [1], "d2": [2], "d3": [10], "d4": [31], "d5": [40], "d6": [60]}
    )
    row = build_features(df).iloc[0]
    assert (row["pares"], row["impares"]) == (4, 2)
    assert row["soma"] == 144
    assert row["amplitude"] == 59
    assert [row[f"q{i}"] for i in range(1, 5)] == [2, 1, 1, 2]


def test_cumulative_frequencies():
    draws = np.array([[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]])
    cumulative = cumulative_frequencies(draws)
    assert cumulative.shape == (2, 60)
    assert cumulative.loc[1, 1] == 2
    assert cumulative.loc[1, 60] == 0
    assert cumulative.iloc[-1].sum() == 12
