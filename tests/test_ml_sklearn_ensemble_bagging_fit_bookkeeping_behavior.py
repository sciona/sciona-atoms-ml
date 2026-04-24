from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def test_bagging_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_fit_bookkeeping import (
        bagging_additional_estimator_count,
        bagging_fit_require_bootstrap_for_oob,
        bagging_fit_require_no_warm_start_with_oob,
        bagging_resolve_max_features,
        bagging_resolve_max_samples,
    )

    assert callable(bagging_resolve_max_samples)
    assert callable(bagging_resolve_max_features)
    assert callable(bagging_fit_require_bootstrap_for_oob)
    assert callable(bagging_fit_require_no_warm_start_with_oob)
    assert callable(bagging_additional_estimator_count)


def test_bagging_resolve_max_samples_matches_private_fit_state() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_fit_bookkeeping import bagging_resolve_max_samples

    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=1,
        max_samples=0.45,
        random_state=3,
    )
    clf.fit(X, y)

    result = bagging_resolve_max_samples(None, clf.max_samples, X.shape[0])
    assert result == clf._max_samples


def test_bagging_resolve_max_features_matches_private_fit_state() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_fit_bookkeeping import bagging_resolve_max_features

    X, y = make_classification(
        n_samples=45,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        random_state=11,
    )
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=1,
        max_features=0.4,
        random_state=5,
    )
    clf.fit(X, y)

    result = bagging_resolve_max_features(clf.max_features, X.shape[1])
    assert result == clf._max_features


def test_bagging_oob_preflight_atoms_match_sklearn_errors() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_fit_bookkeeping import (
        bagging_fit_require_bootstrap_for_oob,
        bagging_fit_require_no_warm_start_with_oob,
    )

    X, y = make_classification(
        n_samples=30,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=13,
    )

    assert bagging_fit_require_bootstrap_for_oob(True, False) is True
    assert bagging_fit_require_no_warm_start_with_oob(False, True) is True

    with pytest.raises(ViolationError):
        bagging_fit_require_bootstrap_for_oob(False, True)
    with pytest.raises(ValueError, match="bootstrap=True"):
        BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=0),
            n_estimators=2,
            bootstrap=False,
            oob_score=True,
            random_state=0,
        ).fit(X, y)

    with pytest.raises(ViolationError):
        bagging_fit_require_no_warm_start_with_oob(True, True)
    with pytest.raises(ValueError, match="warm_start=False"):
        BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=0),
            n_estimators=2,
            bootstrap=True,
            warm_start=True,
            oob_score=True,
            random_state=0,
        ).fit(X, y)


def test_bagging_additional_estimator_count_matches_warm_start_state() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_fit_bookkeeping import bagging_additional_estimator_count

    X, y = make_classification(
        n_samples=35,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=17,
    )
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=2,
        warm_start=True,
        random_state=7,
    )
    clf.fit(X, y)
    clf.n_estimators = 5

    assert bagging_additional_estimator_count(clf.n_estimators, len(clf.estimators_)) == 3


def test_contracts_reject_invalid_bagging_fit_bookkeeping_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_fit_bookkeeping import (
        bagging_additional_estimator_count,
        bagging_resolve_max_features,
        bagging_resolve_max_samples,
    )

    with pytest.raises(ViolationError):
        bagging_resolve_max_samples(None, 1.5, 10)

    with pytest.raises(ViolationError):
        bagging_resolve_max_features(2.0, 4)

    with pytest.raises(ViolationError):
        bagging_additional_estimator_count(2, 3)
