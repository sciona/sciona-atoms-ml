from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.inspection._permutation_importance import _weights_scorer


def test_permutation_weighted_scorer_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_weighted_scorer_shell import (
        permutation_importance_scorer_kwargs,
        permutation_importance_use_sample_weight,
    )

    assert callable(permutation_importance_use_sample_weight)
    assert callable(permutation_importance_scorer_kwargs)


def test_permutation_weighted_scorer_shell_matches_sklearn_branch_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_weighted_scorer_shell import (
        permutation_importance_scorer_kwargs,
        permutation_importance_use_sample_weight,
    )

    observed_kwargs: list[dict[str, object]] = []

    def scorer(estimator, X, y, **kwargs):  # type: ignore[no-untyped-def]
        del estimator
        del X
        del y
        observed_kwargs.append(dict(kwargs))
        return 1.0

    X = np.asarray([[0.0], [1.0]], dtype=np.float64)
    y = np.asarray([0, 1], dtype=np.int64)
    sample_weight = [0.25, 0.75]

    _weights_scorer(scorer, object(), X, y, None)
    _weights_scorer(scorer, object(), X, y, sample_weight)

    assert permutation_importance_use_sample_weight(None) is False
    assert permutation_importance_scorer_kwargs(None) == {}

    assert permutation_importance_use_sample_weight(sample_weight) is True
    actual_kwargs = permutation_importance_scorer_kwargs(sample_weight)
    assert set(actual_kwargs) == {"sample_weight"}
    assert np.array_equal(actual_kwargs["sample_weight"], np.asarray(sample_weight, dtype=np.float64))

    assert observed_kwargs[0] == {}
    assert set(observed_kwargs[1]) == {"sample_weight"}
    assert np.array_equal(
        np.asarray(observed_kwargs[1]["sample_weight"], dtype=np.float64),
        np.asarray(sample_weight, dtype=np.float64),
    )


def test_permutation_weighted_scorer_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_weighted_scorer_shell import (
        permutation_importance_scorer_kwargs,
        permutation_importance_use_sample_weight,
    )

    with pytest.raises(ViolationError):
        permutation_importance_use_sample_weight(1.0)

    with pytest.raises(ViolationError):
        permutation_importance_scorer_kwargs(np.asarray([[1.0, 2.0]], dtype=np.float64))
