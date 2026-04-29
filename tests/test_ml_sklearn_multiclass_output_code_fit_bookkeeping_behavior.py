from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.multiclass.output_code_fit_bookkeeping import (
    output_code_fit_estimator_count,
    output_code_fit_feature_names_in,
    output_code_fit_n_features_in,
    output_code_fit_require_nonempty_classes,
)


def test_output_code_fit_require_nonempty_classes_matches_guard() -> None:
    classes = np.array([1.0, 2.0], dtype=np.float64)
    assert np.array_equal(output_code_fit_require_nonempty_classes(classes), classes)

    with pytest.raises(ValueError, match="no class is present"):
        output_code_fit_require_nonempty_classes(np.array([], dtype=np.float64))


def test_output_code_fit_estimator_count_matches_sklearn_formula() -> None:
    assert output_code_fit_estimator_count(3, 1.5) == 4
    assert output_code_fit_estimator_count(4, 0.5) == 2


def test_output_code_fit_postfit_attribute_pass_through() -> None:
    assert output_code_fit_n_features_in(6) == 6
    assert output_code_fit_feature_names_in(("x0", "x1")) == ("x0", "x1")


def test_output_code_fit_bookkeeping_rejects_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        output_code_fit_estimator_count(0, 1.0)

    with pytest.raises((ViolationError, ValueError)):
        output_code_fit_feature_names_in(("x0", ""))
