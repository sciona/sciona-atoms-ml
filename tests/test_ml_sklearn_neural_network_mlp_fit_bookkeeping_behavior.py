from __future__ import annotations

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor


def test_mlp_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping import (
        mlp_batch_size,
        mlp_batch_size_warning_required,
        mlp_first_pass_required,
        mlp_hidden_layer_sizes,
        mlp_partial_fit_require_no_early_stopping,
    )

    assert callable(mlp_hidden_layer_sizes)
    assert callable(mlp_first_pass_required)
    assert callable(mlp_partial_fit_require_no_early_stopping)
    assert callable(mlp_batch_size_warning_required)
    assert callable(mlp_batch_size)


def test_mlp_hidden_layer_sizes_matches_source_normalization() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping import mlp_hidden_layer_sizes

    assert mlp_hidden_layer_sizes(5) == (5,)
    assert mlp_hidden_layer_sizes((4, 3)) == (4, 3)
    assert mlp_hidden_layer_sizes([2, 1]) == (2, 1)


def test_mlp_hidden_layer_sizes_invalid_case_matches_private_fit_error() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping import mlp_hidden_layer_sizes

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)

    with pytest.raises(ValueError, match="hidden_layer_sizes must be > 0"):
        mlp_hidden_layer_sizes((3, 0))
    with pytest.raises(ValueError, match="hidden_layer_sizes must be > 0"):
        MLPRegressor(hidden_layer_sizes=(3, 0), max_iter=1, random_state=0)._fit(X, y, incremental=False)


def test_mlp_first_pass_required_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping import mlp_first_pass_required

    assert mlp_first_pass_required(has_coefs=False, warm_start=False, incremental=False) is True
    assert mlp_first_pass_required(has_coefs=True, warm_start=False, incremental=False) is True
    assert mlp_first_pass_required(has_coefs=True, warm_start=True, incremental=False) is False
    assert mlp_first_pass_required(has_coefs=True, warm_start=False, incremental=True) is False


def test_mlp_partial_fit_early_stopping_guard_matches_public_contract() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping import mlp_partial_fit_require_no_early_stopping

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    assert mlp_partial_fit_require_no_early_stopping(early_stopping=False, incremental=True) is True
    with pytest.raises(ValueError, match="partial_fit does not support early_stopping=True"):
        mlp_partial_fit_require_no_early_stopping(early_stopping=True, incremental=True)
    with pytest.raises(ValueError, match="partial_fit does not support early_stopping=True"):
        MLPClassifier(early_stopping=True, max_iter=1, random_state=0).partial_fit(X, y, classes=np.array([0, 1]))


def test_mlp_batch_size_resolution_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping import (
        mlp_batch_size,
        mlp_batch_size_warning_required,
    )

    assert mlp_batch_size("auto", n_samples=50) == 50
    assert mlp_batch_size("auto", n_samples=400) == 200
    assert mlp_batch_size(32, n_samples=10) == 10
    assert mlp_batch_size(8, n_samples=10) == 8
    assert mlp_batch_size_warning_required("auto", n_samples=10) is False
    assert mlp_batch_size_warning_required(8, n_samples=10) is False
    assert mlp_batch_size_warning_required(32, n_samples=10) is True


def test_mlp_fit_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping import (
        mlp_batch_size,
        mlp_batch_size_warning_required,
        mlp_first_pass_required,
        mlp_hidden_layer_sizes,
        mlp_partial_fit_require_no_early_stopping,
    )

    with pytest.raises(Exception):
        mlp_hidden_layer_sizes(())
    with pytest.raises(Exception):
        mlp_first_pass_required(has_coefs=np.bool_(True), warm_start=False, incremental=False)
    with pytest.raises(Exception):
        mlp_partial_fit_require_no_early_stopping(early_stopping=False, incremental=np.bool_(True))
    with pytest.raises(Exception):
        mlp_batch_size_warning_required(0, n_samples=10)
    with pytest.raises(Exception):
        mlp_batch_size("large", n_samples=10)
