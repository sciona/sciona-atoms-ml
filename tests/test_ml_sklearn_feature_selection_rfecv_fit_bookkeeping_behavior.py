from __future__ import annotations

import warnings

import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR


def test_rfecv_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping import (
        rfecv_default_scoring_name,
        rfecv_resolved_min_features_to_select,
        rfecv_warn_min_features_too_large,
    )

    assert callable(rfecv_warn_min_features_too_large)
    assert callable(rfecv_resolved_min_features_to_select)
    assert callable(rfecv_default_scoring_name)


def test_rfecv_warning_and_resolved_min_features_match_fit_shell() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping import (
        rfecv_resolved_min_features_to_select,
        rfecv_warn_min_features_too_large,
    )

    X, y = load_iris(return_X_y=True)
    X = X[:, :3]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RFECV(
            estimator=LogisticRegression(max_iter=500),
            min_features_to_select=5,
            cv=2,
        ).fit(X, y)

    assert any("min_features_to_select=5" in str(item.message) for item in caught)
    assert rfecv_warn_min_features_too_large(X.shape[1], min_features_to_select=5) is True
    assert rfecv_resolved_min_features_to_select(X.shape[1], min_features_to_select=5) == X.shape[1]
    assert rfecv_warn_min_features_too_large(X.shape[1], min_features_to_select=2) is False
    assert rfecv_resolved_min_features_to_select(X.shape[1], min_features_to_select=2) == 2


def test_rfecv_default_scoring_name_matches_private_default_logic() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping import (
        rfecv_default_scoring_name,
    )

    clf = RFECV(estimator=LogisticRegression(max_iter=500))
    reg = RFECV(estimator=SVR())

    assert rfecv_default_scoring_name(True) == "accuracy"
    assert rfecv_default_scoring_name(False) == "r2"
    assert rfecv_default_scoring_name(True, "f1_macro") == "f1_macro"

    assert clf._get_scorer()._score_func.__name__ == "accuracy_score"
    assert reg._get_scorer()._score_func.__name__ == "r2_score"


def test_rfecv_fit_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping import (
        rfecv_default_scoring_name,
        rfecv_resolved_min_features_to_select,
        rfecv_warn_min_features_too_large,
    )

    with pytest.raises(ViolationError):
        rfecv_warn_min_features_too_large(1, min_features_to_select=1)

    with pytest.raises(ViolationError):
        rfecv_resolved_min_features_to_select(3, min_features_to_select=0)

    with pytest.raises(ViolationError):
        rfecv_default_scoring_name(True, "")
