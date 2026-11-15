import numpy as np
import pytest

from mega_sena.distances import (
    consecutive_pairs,
    consecutive_pairs_pmf,
    gap_pmf,
    longest_run,
    longest_run_pmf,
    min_gap,
    min_gap_pmf,
    nearest_neighbor,
    nearest_neighbor_pmf,
    nearest_theory,
    neighbor_offsets,
    pair_distance_pmf,
    pair_distances,
    sorted_gaps,
)
from mega_sena.games import MEGA_SENA
from mega_sena.hypothesis import random_draws

GAP_CAP = MEGA_SENA.teto_salto

EXAMPLE = np.array([[4, 5, 30, 33, 41, 52], [1, 2, 3, 10, 20, 60]])


@pytest.fixture(scope="module")
def simulated():
    """200 mil concursos sob H0, para conferir as fórmulas exatas."""
    return random_draws(200_000, np.random.default_rng(0))


def test_metrics_on_known_draws():
    assert sorted_gaps(EXAMPLE).tolist() == [[1, 25, 3, 8, 11], [1, 1, 7, 10, 40]]
    assert consecutive_pairs(EXAMPLE).tolist() == [1, 2]
    assert longest_run(EXAMPLE).tolist() == [2, 3]
    assert min_gap(EXAMPLE).tolist() == [1, 1]
    assert pair_distances(EXAMPLE).shape == (2, 15)
    assert nearest_neighbor(EXAMPLE)[0].tolist() == [1, 1, 3, 3, 8, 11]
    assert sorted(neighbor_offsets(EXAMPLE, 30).tolist()) == [-26, -25, 3, 11, 22]


@pytest.mark.parametrize(
    "pmf",
    [consecutive_pairs_pmf(), longest_run_pmf(), min_gap_pmf(), gap_pmf(), pair_distance_pmf()],
)
def test_pmfs_sum_to_one(pmf):
    assert np.isclose(pmf.sum(), 1)


def test_at_least_one_consecutive_pair_is_about_42_percent():
    assert np.isclose(1 - consecutive_pairs_pmf()[0], 0.421, atol=0.001)


def test_exact_pmfs_match_simulation(simulated):
    tol = 0.004
    consecutive = np.bincount(consecutive_pairs(simulated), minlength=6) / len(simulated)
    assert np.allclose(consecutive, consecutive_pairs_pmf(), atol=tol)

    gaps = np.bincount(np.minimum(sorted_gaps(simulated), GAP_CAP).ravel(), minlength=GAP_CAP + 1)
    assert np.allclose(gaps[1:] / gaps.sum(), gap_pmf(), atol=tol)

    runs = np.bincount(longest_run(simulated), minlength=7)[1:] / len(simulated)
    assert np.allclose(runs, longest_run_pmf(), atol=tol)

    smallest = np.bincount(min_gap(simulated), minlength=len(min_gap_pmf()) + 1)[1:]
    assert np.allclose(smallest / len(simulated), min_gap_pmf(), atol=tol)

    pairs = np.bincount(pair_distances(simulated).ravel(), minlength=60)[1:]
    assert np.allclose(pairs / pairs.sum(), pair_distance_pmf(), atol=tol)


@pytest.mark.parametrize("number", [1, 10, 30, 60])
def test_nearest_neighbor_pmf_matches_simulation(simulated, number):
    pmf = nearest_neighbor_pmf(number)
    assert np.isclose(pmf.sum(), 1)
    rows = (simulated == number).any(axis=1)
    observed = nearest_neighbor(simulated[rows])[simulated[rows] == number]
    assert abs(observed.mean() - nearest_theory().loc[number, "esperado"]) < 0.1


def test_edge_numbers_have_farther_neighbors():
    theory = nearest_theory()
    assert theory.loc[1, "esperado"] > theory.loc[30, "esperado"]
    assert theory.loc[60, "esperado"] > theory.loc[30, "esperado"]
