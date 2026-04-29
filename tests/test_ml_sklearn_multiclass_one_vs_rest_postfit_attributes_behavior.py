from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postfit_attributes import (
    one_vs_rest_fit_classes,
    one_vs_rest_fit_feature_names_in,
    one_vs_rest_fit_n_features_in,
)


def test_one_vs_rest_fit_classes_preserves_label_binarizer_classes() -> None:
    classes = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    observed = one_vs_rest_fit_classes(classes)
    assert np.array_equal(observed, classes)


def test_one_vs_rest_fit_postfit_attribute_pass_through() -> None:
    assert one_vs_rest_fit_n_features_in(5) == 5
    assert one_vs_rest_fit_feature_names_in(("x0", "x1")) == ("x0", "x1")


def test_one_vs_rest_postfit_attributes_reject_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        one_vs_rest_fit_classes(np.array([1.0, 1.0], dtype=np.float64))

    with pytest.raises((ViolationError, ValueError)):
        one_vs_rest_fit_n_features_in(0)

    with pytest.raises((ViolationError, ValueError)):
        one_vs_rest_fit_feature_names_in(("x0", ""))
