from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.multioutput.fit_bookkeeping import (
    multioutput_fit_output_count,
    multioutput_fit_require_2d_targets,
    multioutput_fit_require_base_fit_method,
    multioutput_fit_require_sample_weight_support,
    multioutput_fit_target_column,
)


def test_multioutput_fit_require_base_fit_method_matches_guard() -> None:
    assert multioutput_fit_require_base_fit_method(True) is True
    with pytest.raises(ValueError, match="fit method"):
        multioutput_fit_require_base_fit_method(False)


def test_multioutput_fit_require_2d_targets_matches_multioutput_fit_guard() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    observed = multioutput_fit_require_2d_targets(y)
    assert np.array_equal(observed, y)

    with pytest.raises(ValueError, match="at least two dimensions"):
        multioutput_fit_require_2d_targets(np.array([1.0, 2.0], dtype=np.float64))


def test_multioutput_fit_output_count_and_target_column_match_sklearn_loop_shape() -> None:
    y = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]], dtype=np.float64)

    assert multioutput_fit_output_count(y) == 3
    assert np.array_equal(multioutput_fit_target_column(y, 1), y[:, 1])


def test_multioutput_fit_require_sample_weight_support_matches_guard() -> None:
    assert multioutput_fit_require_sample_weight_support(
        sample_weight_provided=False,
        estimator_supports_sample_weight=False,
    ) is True
    assert multioutput_fit_require_sample_weight_support(
        sample_weight_provided=True,
        estimator_supports_sample_weight=True,
    ) is True

    with pytest.raises(ValueError, match="does not support sample weights"):
        multioutput_fit_require_sample_weight_support(
            sample_weight_provided=True,
            estimator_supports_sample_weight=False,
        )


def test_multioutput_fit_bookkeeping_rejects_invalid_inputs() -> None:
    with pytest.raises((ValueError, ViolationError)):
        multioutput_fit_output_count(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises((ValueError, ViolationError)):
        multioutput_fit_target_column(np.array([[1.0, 2.0]], dtype=np.float64), 2)
