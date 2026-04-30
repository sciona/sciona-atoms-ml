from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def test_rfe_postfit_attributes_imports() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_postfit_attributes import (
        rfe_classes,
        rfe_estimator_type,
        rfe_support_mask,
    )

    assert callable(rfe_estimator_type)
    assert callable(rfe_classes)
    assert callable(rfe_support_mask)


def test_rfe_postfit_attributes_match_fitted_classifier() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_postfit_attributes import (
        rfe_classes,
        rfe_estimator_type,
        rfe_support_mask,
    )

    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    selector = RFE(LogisticRegression(max_iter=1000, random_state=0), n_features_to_select=3)
    selector.fit(X, y)

    assert rfe_estimator_type(selector.estimator._estimator_type) == selector._estimator_type
    assert rfe_classes(tuple(selector.estimator_.classes_.tolist())) == tuple(selector.classes_.tolist())
    assert np.array_equal(rfe_support_mask(selector.support_), selector._get_support_mask())


def test_rfe_postfit_attributes_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_postfit_attributes import (
        rfe_classes,
        rfe_estimator_type,
        rfe_support_mask,
    )

    with pytest.raises(ViolationError):
        rfe_estimator_type("")

    with pytest.raises(ViolationError):
        rfe_classes(())

    with pytest.raises(ViolationError):
        rfe_support_mask(np.array([1, 0, 1], dtype=np.int64))  # type: ignore[arg-type]
