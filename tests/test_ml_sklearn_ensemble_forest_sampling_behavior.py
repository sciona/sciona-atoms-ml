from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.ensemble._forest import (
    _generate_sample_indices,
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
)


def test_forest_sampling_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_sampling import (
        forest_generate_sample_indices,
        forest_generate_unsampled_indices,
        forest_resolve_bootstrap_sample_count,
    )

    assert callable(forest_resolve_bootstrap_sample_count)
    assert callable(forest_generate_sample_indices)
    assert callable(forest_generate_unsampled_indices)


@pytest.mark.parametrize("max_samples", [None, 7, 0.2, 0.75, 1.0])
def test_forest_resolve_bootstrap_sample_count_matches_sklearn(max_samples: int | float | None) -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_sampling import forest_resolve_bootstrap_sample_count

    assert forest_resolve_bootstrap_sample_count(11, max_samples) == _get_n_samples_bootstrap(11, max_samples)


def test_forest_generate_sample_indices_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_sampling import forest_generate_sample_indices

    result = forest_generate_sample_indices(9, 14, random_state=23)
    expected = _generate_sample_indices(np.random.RandomState(23), 9, 14)
    assert np.array_equal(result, expected)


def test_forest_generate_unsampled_indices_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_sampling import forest_generate_unsampled_indices

    result = forest_generate_unsampled_indices(12, 9, random_state=31)
    expected = _generate_unsampled_indices(np.random.RandomState(31), 12, 9)
    assert np.array_equal(result, expected)


def test_unsampled_indices_are_complement_of_sampled_indices_for_same_seed() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_sampling import (
        forest_generate_sample_indices,
        forest_generate_unsampled_indices,
    )

    sampled = forest_generate_sample_indices(10, 8, random_state=19)
    unsampled = forest_generate_unsampled_indices(10, 8, random_state=19)
    sampled_set = set(np.unique(sampled).tolist())
    unsampled_set = set(unsampled.tolist())
    assert sampled_set.isdisjoint(unsampled_set)
    assert sampled_set | unsampled_set == set(range(10))


def test_contracts_reject_invalid_forest_sampling_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_sampling import (
        forest_generate_sample_indices,
        forest_generate_unsampled_indices,
        forest_resolve_bootstrap_sample_count,
    )

    with pytest.raises(ViolationError):
        forest_resolve_bootstrap_sample_count(0, None)

    with pytest.raises(ViolationError):
        forest_resolve_bootstrap_sample_count(5, 6)

    with pytest.raises(ViolationError):
        forest_resolve_bootstrap_sample_count(5, 0.0)

    with pytest.raises(ViolationError):
        forest_resolve_bootstrap_sample_count(5, 1.1)

    with pytest.raises(ViolationError):
        forest_generate_sample_indices(0, 4, random_state=0)

    with pytest.raises(ViolationError):
        forest_generate_unsampled_indices(5, 0, random_state=0)
