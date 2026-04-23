from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.ensemble._bagging import _generate_bagging_indices, _generate_indices


def test_bagging_sampling_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_sampling import (
        bagging_generate_bagging_indices,
        bagging_generate_indices,
    )

    assert callable(bagging_generate_indices)
    assert callable(bagging_generate_bagging_indices)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_bagging_generate_indices_matches_sklearn(bootstrap: bool) -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_sampling import bagging_generate_indices

    result = bagging_generate_indices(bootstrap, 7, 4, random_state=13)
    expected = _generate_indices(np.random.RandomState(13), bootstrap, 7, 4)
    assert np.array_equal(result, expected)


def test_bagging_generate_indices_bootstrap_allows_repeated_indices() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_sampling import bagging_generate_indices

    result = bagging_generate_indices(True, 3, 9, random_state=5)
    assert result.shape == (9,)
    assert np.unique(result).shape[0] < result.shape[0]


@pytest.mark.parametrize(
    ("bootstrap_features", "bootstrap_samples", "sample_weight"),
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (True, True, None),
        (False, True, np.array([1.0, 2.0, 1.0, 1.5, 0.5, 3.0, 2.5, 1.0, 2.0, 1.5], dtype=np.float64)),
    ],
)
def test_bagging_generate_bagging_indices_matches_sklearn(
    bootstrap_features: bool,
    bootstrap_samples: bool,
    sample_weight: np.ndarray | None,
) -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_sampling import bagging_generate_bagging_indices

    feature_indices, sample_indices = bagging_generate_bagging_indices(
        bootstrap_features,
        bootstrap_samples,
        8,
        10,
        5,
        6,
        random_state=17,
        sample_weight=sample_weight,
    )
    expected_features, expected_samples = _generate_bagging_indices(
        17,
        bootstrap_features,
        bootstrap_samples,
        8,
        10,
        5,
        6,
        sample_weight,
    )
    assert np.array_equal(feature_indices, expected_features)
    assert np.array_equal(sample_indices, expected_samples)


def test_contracts_reject_invalid_bagging_sampling_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_sampling import (
        bagging_generate_bagging_indices,
        bagging_generate_indices,
    )

    with pytest.raises(ViolationError):
        bagging_generate_indices(False, 4, 5, random_state=0)

    with pytest.raises(ViolationError):
        bagging_generate_indices(True, 0, 3, random_state=0)

    with pytest.raises(ViolationError):
        bagging_generate_bagging_indices(False, True, 4, 5, 6, 3, random_state=0)

    with pytest.raises(ViolationError):
        bagging_generate_bagging_indices(True, False, 4, 5, 3, 7, random_state=0)

    with pytest.raises(ViolationError):
        bagging_generate_bagging_indices(
            True,
            True,
            4,
            5,
            3,
            4,
            random_state=0,
            sample_weight=np.array([1.0, -1.0, 2.0, 1.0, 1.0], dtype=np.float64),
        )
