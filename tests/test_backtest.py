import numpy as np

from mega_sena.hypothesis import random_draws
from mega_sena.modeling.backtest import (
    EXPECTED_HITS,
    cold_strategy,
    count_hits,
    hot_strategy,
    random_strategy,
    run_backtest,
)


def test_count_hits():
    bets = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    draws = np.array([[1, 2, 3, 40, 50, 60], [7, 8, 9, 10, 11, 12]])
    assert count_hits(bets, draws).tolist() == [3, 0]


def test_hot_and_cold_pick_extremes():
    past = np.array([[1, 2, 3, 4, 5, 6]] * 3 + [[7, 8, 9, 10, 11, 12]])
    rng = np.random.default_rng(0)
    assert hot_strategy()(past, rng).tolist() == [1, 2, 3, 4, 5, 6]
    assert not set(cold_strategy()(past, rng)) & set(range(1, 13))
    assert hot_strategy(window=1)(past, rng).tolist() == [7, 8, 9, 10, 11, 12]


def test_random_strategy_is_valid_bet():
    bet = random_strategy(np.empty((0, 6)), np.random.default_rng(0))
    assert len(set(bet)) == 6 and bet.min() >= 1 and bet.max() <= 60


def test_backtest_has_no_lookahead():
    """A estratégia só pode ver os concursos anteriores ao que está apostando."""
    draws = random_draws(50, np.random.default_rng(0))
    seen_lengths = []

    def spy(past, rng):
        seen_lengths.append(len(past))
        return random_strategy(past, rng)

    run_backtest(draws, spy, start=10)
    assert seen_lengths == list(range(10, 50))


def test_hot_strategy_matches_chance_on_random_data():
    draws = random_draws(3000, np.random.default_rng(2))
    hits = run_backtest(draws, hot_strategy(), start=100)
    assert abs(hits.mean() - EXPECTED_HITS) < 0.05
