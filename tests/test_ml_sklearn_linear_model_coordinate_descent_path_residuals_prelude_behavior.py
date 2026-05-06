from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_path_residuals_prelude_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_prelude import (
        cd_path_residuals_rescaled_train_sample_weight,
        cd_path_residuals_resolved_precompute,
        cd_path_residuals_test_sample_weight,
        cd_path_residuals_train_sample_count,
        cd_path_residuals_train_sample_weight,
        cd_path_residuals_use_gram_precompute,
        cd_path_residuals_use_sample_weight_branch,
    )

    assert callable(cd_path_residuals_use_sample_weight_branch)
    assert callable(cd_path_residuals_train_sample_weight)
    assert callable(cd_path_residuals_test_sample_weight)
    assert callable(cd_path_residuals_train_sample_count)
    assert callable(cd_path_residuals_rescaled_train_sample_weight)
    assert callable(cd_path_residuals_use_gram_precompute)
    assert callable(cd_path_residuals_resolved_precompute)


def test_coordinate_descent_path_residuals_prelude_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_prelude import (
        cd_path_residuals_rescaled_train_sample_weight,
        cd_path_residuals_resolved_precompute,
        cd_path_residuals_test_sample_weight,
        cd_path_residuals_train_sample_count,
        cd_path_residuals_train_sample_weight,
        cd_path_residuals_use_gram_precompute,
        cd_path_residuals_use_sample_weight_branch,
    )

    sample_weight = np.array([1.0, 2.0, 3.0, 4.0])
    train = np.array([0, 2, 3])
    test = np.array([1])
    sw_train = cd_path_residuals_train_sample_weight(sample_weight, train)
    sw_test = cd_path_residuals_test_sample_weight(sample_weight, test)
    assert cd_path_residuals_use_sample_weight_branch(sample_weight) is True
    assert np.array_equal(sw_train, np.array([1.0, 3.0, 4.0]))
    assert np.array_equal(sw_test, np.array([2.0]))
    n_samples = cd_path_residuals_train_sample_count((3, 5))
    assert n_samples == 3
    rescaled = cd_path_residuals_rescaled_train_sample_weight(sw_train, n_samples)
    assert np.isclose(np.sum(rescaled), 3.0)
    assert cd_path_residuals_use_gram_precompute(1) is True
    assert cd_path_residuals_resolved_precompute("auto", True) == "auto"
    assert cd_path_residuals_resolved_precompute("auto", False) is False


def test_coordinate_descent_path_residuals_prelude_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_prelude import (
        cd_path_residuals_rescaled_train_sample_weight,
        cd_path_residuals_use_gram_precompute,
    )

    with pytest.raises(ViolationError):
        cd_path_residuals_use_gram_precompute(0)

    with pytest.raises(ViolationError):
        cd_path_residuals_rescaled_train_sample_weight(np.array([]), 3)
