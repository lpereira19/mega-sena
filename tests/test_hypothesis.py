import numpy as np

from mega_sena.hypothesis import (
    _merge_sparse_bins,
    category_totals_test,
    hypergeom_pmf,
    random_draws,
)


def test_random_draws_are_valid():
    draws = random_draws(1000, np.random.default_rng(0))
    assert draws.shape == (1000, 6)
    assert draws.min() >= 1 and draws.max() <= 60
    assert all(len(set(row)) == 6 for row in draws)


def test_hypergeom_pmf():
    pmf = hypergeom_pmf(30)
    assert np.isclose(pmf.sum(), 1)
    assert np.isclose(pmf @ np.arange(7), 3)  # média: 6 * 30/60
    assert np.isclose(hypergeom_pmf(6)[6], 1 / 50_063_860)  # chance da sena


def test_merge_sparse_bins_keeps_totals():
    obs, exp = _merge_sparse_bins(np.array([1, 10, 20, 3, 1]), np.array([2, 9, 18, 4, 1]), 5)
    assert obs.sum() == 35 and exp.sum() == 34
    assert (exp >= 5).all()


def test_corrected_chi2_is_calibrated_under_h0():
    """Sob H0, o p-valor deve ser uniforme: ~5% de rejeições a 5% de significância."""
    rng = np.random.default_rng(1)
    p_values = []
    for _ in range(400):
        draws = random_draws(500, rng)
        counts = np.bincount(draws.ravel(), minlength=61)[1:]
        p_values.append(category_totals_test(counts, np.ones(60), 500, "teste").p_valor)
    rejection_rate = np.mean(np.array(p_values) < 0.05)
    assert 0.02 < rejection_rate < 0.09
