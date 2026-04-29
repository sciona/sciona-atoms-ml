from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.multioutput.postfit_attributes import (
    multioutput_classifier_classes,
    multioutput_fit_feature_names_in,
    multioutput_fit_n_features_in,
    multioutput_partial_fit_feature_names_in_update_required,
    multioutput_partial_fit_n_features_in_update_required,
)


def test_multioutput_partial_fit_attribute_update_guards_match_sklearn() -> None:
    assert multioutput_partial_fit_n_features_in_update_required(
        first_time=True,
        estimator_has_n_features_in=True,
    ) is True
    assert multioutput_partial_fit_n_features_in_update_required(
        first_time=False,
        estimator_has_n_features_in=True,
    ) is False
    assert multioutput_partial_fit_n_features_in_update_required(
        first_time=True,
        estimator_has_n_features_in=False,
    ) is False

    assert multioutput_partial_fit_feature_names_in_update_required(
        first_time=True,
        estimator_has_feature_names_in=True,
    ) is True
    assert multioutput_partial_fit_feature_names_in_update_required(
        first_time=False,
        estimator_has_feature_names_in=True,
    ) is False
    assert multioutput_partial_fit_feature_names_in_update_required(
        first_time=True,
        estimator_has_feature_names_in=False,
    ) is False


def test_multioutput_fit_attribute_pass_through_matches_first_estimator_copy() -> None:
    assert multioutput_fit_n_features_in(4) == 4
    assert multioutput_fit_feature_names_in(("x0", "x1")) == ("x0", "x1")


def test_multioutput_classifier_classes_preserves_per_estimator_class_vectors() -> None:
    class_vectors = (
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([1.0, 3.0, 5.0], dtype=np.float64),
    )
    observed = multioutput_classifier_classes(class_vectors)
    assert len(observed) == 2
    assert np.array_equal(observed[0], class_vectors[0])
    assert np.array_equal(observed[1], class_vectors[1])


def test_multioutput_postfit_attributes_reject_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        multioutput_fit_n_features_in(0)

    with pytest.raises((ViolationError, ValueError)):
        multioutput_fit_feature_names_in(("x0", ""))

    with pytest.raises((ViolationError, ValueError)):
        multioutput_classifier_classes((np.array([1.0, 1.0], dtype=np.float64),))
