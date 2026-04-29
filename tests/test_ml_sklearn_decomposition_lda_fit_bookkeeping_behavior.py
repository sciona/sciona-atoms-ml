from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import gen_batches


def test_lda_fit_bookkeeping_imports() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping import (
        lda_batch_bounds,
        lda_check_nonnegative_dtype_names,
        lda_fit_converged,
        lda_fit_evaluate_iteration_due,
        lda_fit_use_online_batches,
        lda_partial_fit_first_call,
        lda_partial_fit_require_matching_feature_count,
    )

    assert callable(lda_fit_use_online_batches)
    assert callable(lda_fit_evaluate_iteration_due)
    assert callable(lda_fit_converged)
    assert callable(lda_batch_bounds)
    assert callable(lda_partial_fit_first_call)
    assert callable(lda_partial_fit_require_matching_feature_count)
    assert callable(lda_check_nonnegative_dtype_names)


def test_lda_fit_bookkeeping_matches_fit_and_partial_fit_shell_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping import (
        lda_fit_converged,
        lda_fit_evaluate_iteration_due,
        lda_fit_use_online_batches,
        lda_partial_fit_first_call,
    )

    assert lda_fit_use_online_batches("online") is True
    assert lda_fit_use_online_batches("batch") is False

    assert lda_fit_evaluate_iteration_due(2, evaluate_every=3) is True
    assert lda_fit_evaluate_iteration_due(1, evaluate_every=3) is False

    assert lda_fit_converged(None, 9.0, perp_tol=0.1) is False
    assert lda_fit_converged(9.0, 8.95, perp_tol=0.1) is True
    assert lda_fit_converged(9.0, 8.7, perp_tol=0.1) is False

    model = LatentDirichletAllocation(n_components=2, random_state=0)
    assert lda_partial_fit_first_call(has_components=hasattr(model, "components_")) is True
    model._init_latent_vars(4, dtype=np.float64)
    assert lda_partial_fit_first_call(has_components=hasattr(model, "components_")) is False


def test_lda_batch_bounds_match_sklearn_gen_batches() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping import lda_batch_bounds

    observed = lda_batch_bounds(11, batch_size=4)
    expected = np.asarray([(sl.start, sl.stop) for sl in gen_batches(11, 4)], dtype=np.int64)

    assert np.array_equal(observed, expected)


def test_lda_partial_fit_feature_count_guard_matches_sklearn_error() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping import lda_partial_fit_require_matching_feature_count

    X = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    model = LatentDirichletAllocation(n_components=2, random_state=0)
    model._init_latent_vars(X.shape[1], dtype=X.dtype)

    assert lda_partial_fit_require_matching_feature_count(3, trained_feature_count=3) == 3

    with pytest.raises(ValueError, match="The provided data has 2 dimensions while the model was trained with feature size 3."):
        lda_partial_fit_require_matching_feature_count(2, trained_feature_count=3)

    with pytest.raises(ValueError, match="The provided data has 2 dimensions while the model was trained with feature size 3."):
        model.partial_fit(X[:, :2])


def test_lda_check_nonnegative_dtype_names_match_private_branch() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping import lda_check_nonnegative_dtype_names

    assert np.array_equal(
        lda_check_nonnegative_dtype_names(reset_n_features=True),
        np.asarray(["float64", "float32"], dtype=object),
    )
    assert np.array_equal(
        lda_check_nonnegative_dtype_names(reset_n_features=False, components_dtype_name="float32"),
        np.asarray(["float32"], dtype=object),
    )


def test_lda_fit_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping import (
        lda_batch_bounds,
        lda_check_nonnegative_dtype_names,
        lda_fit_converged,
        lda_fit_evaluate_iteration_due,
        lda_fit_use_online_batches,
        lda_partial_fit_require_matching_feature_count,
    )

    with pytest.raises(ViolationError):
        lda_fit_use_online_batches("streaming")

    with pytest.raises(ViolationError):
        lda_fit_evaluate_iteration_due(-1, evaluate_every=3)

    with pytest.raises(ViolationError):
        lda_fit_converged(-1.0, 1.0, perp_tol=0.1)

    with pytest.raises(ViolationError):
        lda_batch_bounds(0, batch_size=2)

    with pytest.raises(ViolationError):
        lda_partial_fit_require_matching_feature_count(0, trained_feature_count=3)

    with pytest.raises(ViolationError):
        lda_check_nonnegative_dtype_names(reset_n_features=False, components_dtype_name="float16")
