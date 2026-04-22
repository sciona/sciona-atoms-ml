from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse


def test_diagnostic_atoms_import() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import (
        check_lasso_sample_complexity,
        check_time_series_index,
        compute_condition_number,
        compute_dispersion_index,
        compute_excess_kurtosis,
        compute_explained_variance_ratio,
        compute_mutual_incoherence,
        compute_n_p_ratio,
        compute_residual_kurtosis,
        compute_skewness,
        compute_vif,
        count_categorical_features,
        estimate_noise_level,
        estimate_tweedie_power,
        is_sparse,
        test_normality,
    )
    assert callable(compute_condition_number)
    assert callable(test_normality)


def test_condition_number_identity_is_one() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_condition_number

    X = np.eye(5, dtype=np.float64)
    assert compute_condition_number(X) == pytest.approx(1.0)


def test_condition_number_ill_conditioned_matrix() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_condition_number

    X = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]], dtype=np.float64)
    assert compute_condition_number(X) > 30


def test_n_p_ratio() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_n_p_ratio

    X = np.zeros((100, 10), dtype=np.float64)
    assert compute_n_p_ratio(X) == pytest.approx(10.0)


def test_mutual_incoherence_orthogonal_is_zero() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_mutual_incoherence

    X = np.eye(5, dtype=np.float64)
    assert compute_mutual_incoherence(X) == pytest.approx(0.0)


def test_lasso_sample_complexity_sufficient() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import check_lasso_sample_complexity

    X = np.zeros((1000, 50), dtype=np.float64)
    assert check_lasso_sample_complexity(X, sparsity_estimate=5) is True


def test_lasso_sample_complexity_insufficient() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import check_lasso_sample_complexity

    X = np.zeros((10, 50), dtype=np.float64)
    assert check_lasso_sample_complexity(X, sparsity_estimate=5) is False


def test_excess_kurtosis_gaussian_near_zero() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_excess_kurtosis

    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, size=10000)
    kurt = compute_excess_kurtosis(y)
    assert abs(kurt) < 0.5


def test_residual_kurtosis_clean_data_is_low() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_residual_kurtosis

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(200, 3))
    y = X @ np.array([1.0, 2.0, 3.0]) + rng.normal(0, 0.1, size=200)
    kurt = compute_residual_kurtosis(X, y)
    assert abs(kurt) < 2.0


def test_dispersion_index_poisson_data() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_dispersion_index

    rng = np.random.default_rng(42)
    y = rng.poisson(lam=5.0, size=10000).astype(np.float64)
    di = compute_dispersion_index(y)
    assert abs(di - 1.0) < 0.2


def test_tweedie_power_poisson_data() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import estimate_tweedie_power

    rng = np.random.default_rng(42)
    y = rng.poisson(lam=10.0, size=5000).astype(np.float64)
    power = estimate_tweedie_power(y)
    assert 0.0 <= power <= 3.0  # Tweedie power is clipped to [0, 3]


def test_noise_level_low_for_clean_signal() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import estimate_noise_level

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(200, 3))
    y = X[:, 0] * 5.0
    noise = estimate_noise_level(X, y)
    assert noise < 1.0


def test_count_categorical_features() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import count_categorical_features

    X = np.column_stack([
        np.array([1, 2, 3, 1, 2, 3], dtype=np.float64),
        np.random.default_rng(42).normal(0, 1, size=6),
    ])
    assert count_categorical_features(X) == 1


def test_skewness_symmetric_is_near_zero() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_skewness

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=10000)
    assert abs(compute_skewness(x)) < 0.2


def test_vif_independent_features() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_vif

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(200, 3))
    vif = compute_vif(X)
    assert all(v >= 1.0 for v in vif)
    assert all(v < 3.0 for v in vif)


def test_normality_gaussian_high_pvalue() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import test_normality

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=100)
    assert test_normality(x) > 0.05


def test_is_sparse_detection() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import is_sparse as check_sparse

    X_dense = np.eye(5, dtype=np.float64)
    X_sparse = sparse.csr_matrix(X_dense)
    assert check_sparse(X_dense) is False
    assert check_sparse(X_sparse) is True


def test_explained_variance_ratio_full_rank() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import compute_explained_variance_ratio

    X = np.eye(5, dtype=np.float64)
    X = np.vstack([X, X])  # 10 x 5
    ratio = compute_explained_variance_ratio(X, n_components=5)
    assert ratio == pytest.approx(1.0)


def test_time_series_index_monotonic() -> None:
    from sciona.atoms.ml.model_selection.diagnostics.atoms import check_time_series_index

    X = np.column_stack([
        np.arange(10, dtype=np.float64),
        np.random.default_rng(42).normal(0, 1, size=10),
    ])
    assert check_time_series_index(X) is True

    X_shuffled = X.copy()
    X_shuffled[:, 0] = X_shuffled[::-1, 0]
    assert check_time_series_index(X_shuffled) is False
