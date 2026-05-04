from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.covariance import MinCovDet
from sklearn.utils import check_random_state


def test_mincovdet_fit_prelude_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_fit_prelude import (
        mincovdet_fit_assume_centered_branch,
        mincovdet_fit_random_state,
        mincovdet_fit_shape,
        mincovdet_fit_validated_data,
    )

    assert callable(mincovdet_fit_validated_data)
    assert callable(mincovdet_fit_random_state)
    assert callable(mincovdet_fit_shape)
    assert callable(mincovdet_fit_assume_centered_branch)


def test_mincovdet_fit_prelude_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_fit_prelude import (
        mincovdet_fit_assume_centered_branch,
        mincovdet_fit_random_state,
        mincovdet_fit_shape,
        mincovdet_fit_validated_data,
    )

    X = [[1, 2], [3, 4], [5, 6]]
    checked = mincovdet_fit_validated_data(X)
    assert checked.dtype == np.float64
    assert np.array_equal(checked, np.asarray(X, dtype=np.float64))
    assert mincovdet_fit_shape(checked) == (3, 2)
    rng = mincovdet_fit_random_state(7)
    assert isinstance(rng, np.random.RandomState)
    assert np.array_equal(rng.uniform(size=3), check_random_state(7).uniform(size=3))
    assert mincovdet_fit_assume_centered_branch(True) is True
    assert mincovdet_fit_assume_centered_branch(False) is False


def test_mincovdet_fit_prelude_matches_fit_call_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_fit_prelude import (
        mincovdet_fit_random_state,
        mincovdet_fit_shape,
        mincovdet_fit_validated_data,
    )

    X = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 7.0]], dtype=np.float64)
    model = MinCovDet(random_state=11)
    captured: dict[str, object] = {}

    def _fake_fast_mcd(data, *, support_fraction, cov_computation_method, random_state):
        captured["X"] = np.asarray(data, dtype=np.float64)
        captured["random_state_sample"] = random_state.uniform(size=3)
        n_samples, n_features = data.shape
        return (
            np.zeros(n_features, dtype=np.float64),
            np.eye(n_features, dtype=np.float64),
            np.ones(n_samples, dtype=np.bool_),
            np.zeros(n_samples, dtype=np.float64),
        )

    with (
        patch("sklearn.covariance._robust_covariance.fast_mcd", side_effect=_fake_fast_mcd),
        patch.object(MinCovDet, "correct_covariance", autospec=True, return_value=None),
        patch.object(MinCovDet, "reweight_covariance", autospec=True, return_value=None),
    ):
        model.fit(X)

    checked = mincovdet_fit_validated_data(X)
    assert np.array_equal(captured["X"], checked)
    assert mincovdet_fit_shape(checked) == checked.shape
    rng = mincovdet_fit_random_state(11)
    assert np.allclose(captured["random_state_sample"], rng.uniform(size=3))


def test_mincovdet_fit_prelude_contracts() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_fit_prelude import (
        mincovdet_fit_assume_centered_branch,
        mincovdet_fit_random_state,
        mincovdet_fit_shape,
        mincovdet_fit_validated_data,
    )

    with pytest.raises(Exception):
        mincovdet_fit_validated_data([[1.0, 2.0]])

    with pytest.raises(Exception):
        mincovdet_fit_random_state("seed")  # type: ignore[arg-type]

    with pytest.raises(Exception):
        mincovdet_fit_shape(np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64))

    with pytest.raises(Exception):
        mincovdet_fit_assume_centered_branch(1)  # type: ignore[arg-type]
