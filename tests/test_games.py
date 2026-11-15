"""As análises precisam valer para qualquer jogo, não só para a Mega-Sena.

Aqui usamos especificações fictícias com regras parecidas com as de outras loterias (volante
pequeno com muitas sorteadas, volante começando em 00, aposta maior que o sorteio) e conferimos
as fórmulas exatas contra simulação.
"""

import numpy as np
import pandas as pd
import pytest

from mega_sena import distances
from mega_sena.dataset import build_dataframe
from mega_sena.features import quadrant_of, quadrant_sizes
from mega_sena.games import MEGA_SENA, GameSpec, get_game
from mega_sena.hypothesis import category_totals_test, random_draws
from mega_sena.modeling.backtest import count_hits, expected_hits, hits_pmf, random_strategy

LIKE_LOTOFACIL = GameSpec("teste15", "Teste 15/25", 1, 25, 15, 15, (5, 5), teto_salto=6)
LIKE_LOTOMANIA = GameSpec("teste20", "Teste 20/100", 0, 99, 20, 50, (10, 10), teto_salto=20)
LIKE_QUINA = GameSpec("teste5", "Teste 5/80", 1, 80, 5, 5, (8, 10), teto_salto=40)
SPECS = [MEGA_SENA, LIKE_LOTOFACIL, LIKE_LOTOMANIA, LIKE_QUINA]


def test_registry():
    assert get_game("megasena") is MEGA_SENA
    with pytest.raises(KeyError):
        get_game("inexistente")


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError):
        GameSpec("x", "volante errado", 1, 60, 6, 6, (6, 9))
    with pytest.raises(ValueError):
        GameSpec("x", "aposta pequena", 1, 60, 6, 5, (6, 10))
    with pytest.raises(ValueError):
        GameSpec("x", "teto grande", 1, 25, 15, 15, (5, 5), teto_salto=25)


def test_mega_sena_paths_are_per_game():
    paths = MEGA_SENA.paths
    assert paths.raw.name == "megasena_api.jsonl"
    assert paths.processed.name == "megasena.csv"
    assert paths.figures.name == "megasena"
    assert paths.hypothesis_tests.parent.name == "megasena"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.slug)
def test_random_draws_respect_the_volante(spec):
    draws = random_draws(2000, np.random.default_rng(0), spec)
    assert draws.shape == (2000, spec.sorteadas)
    assert draws.min() >= spec.menor and draws.max() <= spec.maior
    assert all(len(set(row)) == spec.sorteadas for row in draws[:200])
    bets = random_draws(10, np.random.default_rng(0), spec, size=spec.aposta)
    assert bets.shape == (10, spec.aposta)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.slug)
def test_quadrants_cover_the_volante(spec):
    assert quadrant_sizes(spec).sum() == spec.n
    assert set(quadrant_of(spec.numbers, spec)) == {1, 2, 3, 4}


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.slug)
def test_distance_pmfs_match_simulation(spec):
    draws = random_draws(100_000, np.random.default_rng(1), spec)
    n, tol = len(draws), 0.006
    pmfs = [
        distances.consecutive_pairs_pmf(spec),
        distances.longest_run_pmf(spec),
        distances.min_gap_pmf(spec),
        distances.gap_pmf(spec),
        distances.pair_distance_pmf(spec),
    ]
    assert all(np.isclose(pmf.sum(), 1) for pmf in pmfs)

    consecutive = np.bincount(distances.consecutive_pairs(draws), minlength=spec.sorteadas) / n
    assert np.allclose(consecutive, pmfs[0], atol=tol)
    runs = np.bincount(distances.longest_run(draws), minlength=spec.sorteadas + 1)[1:] / n
    assert np.allclose(runs, pmfs[1], atol=tol)
    smallest = np.bincount(distances.min_gap(draws), minlength=len(pmfs[2]) + 1)[1:] / n
    assert np.allclose(smallest, pmfs[2], atol=tol)
    gaps = np.minimum(distances.sorted_gaps(draws), spec.teto_salto).ravel()
    assert np.allclose(
        np.bincount(gaps, minlength=spec.teto_salto + 1)[1:] / gaps.size, pmfs[3], atol=tol
    )


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.slug)
def test_nearest_neighbor_theory_matches_simulation(spec):
    draws = random_draws(50_000, np.random.default_rng(2), spec)
    table = distances.nearest_by_number(draws, spec)
    assert np.isclose(distances.nearest_neighbor_pmf(spec.menor, spec).sum(), 1)
    # médias observadas próximas da teoria em todas as dezenas (|z| raramente passa de 4)
    assert (table["z"].abs() < 4).all()


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.slug)
def test_hits_distribution_with_bets_larger_than_draw(spec):
    rng = np.random.default_rng(3)
    draws = random_draws(50_000, rng, spec)
    bets = np.array([random_strategy(None, rng, spec) for _ in range(len(draws))])
    hits = count_hits(bets, draws)
    assert abs(hits.mean() - expected_hits(spec)) < 0.05 * max(1, expected_hits(spec))
    observed = np.bincount(hits, minlength=spec.sorteadas + 1) / len(hits)
    assert np.allclose(observed, hits_pmf(spec), atol=0.01)


@pytest.mark.parametrize("spec", [LIKE_LOTOFACIL, LIKE_LOTOMANIA], ids=lambda s: s.slug)
def test_corrected_chi2_is_calibrated_for_other_games(spec):
    """Com 15 de 25 a correção (N-K)/(N-1) pesa muito; sem ela o teste ficaria descalibrado."""
    rng = np.random.default_rng(4)
    p_values = []
    for _ in range(300):
        draws = random_draws(300, rng, spec)
        counts = np.bincount(draws.ravel() - spec.menor, minlength=spec.n)
        p_values.append(category_totals_test(counts, np.ones(spec.n), 300, "t", spec).p_valor)
    assert 0.02 < np.mean(np.array(p_values) < 0.05) < 0.09


def test_build_dataframe_handles_zero_based_volante():
    record = {
        "numero": 1,
        "dataApuracao": "02/10/1999",
        "listaDezenas": [f"{x:02d}" for x in range(0, 100, 5)],
        "acumulado": False,
    }
    df = build_dataframe([record], LIKE_LOTOMANIA)
    assert df.loc[0, "d1"] == 0 and df.loc[0, "d20"] == 95
    assert isinstance(df.loc[0, "data"], pd.Timestamp)
