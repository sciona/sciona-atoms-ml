from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import type_of_target


def test_forest_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_fit_bookkeeping import (
        forest_fit_additional_estimator_count,
        forest_fit_bootstrap_sample_count,
        forest_fit_oob_update_required,
        forest_fit_require_bootstrap_for_oob,
        forest_fit_require_supported_oob_target_type,
    )

    assert callable(forest_fit_bootstrap_sample_count)
    assert callable(forest_fit_require_bootstrap_for_oob)
    assert callable(forest_fit_additional_estimator_count)
    assert callable(forest_fit_oob_update_required)
    assert callable(forest_fit_require_supported_oob_target_type)


@pytest.mark.parametrize("max_samples", [None, 7, 0.4])
def test_forest_fit_bootstrap_sample_count_matches_sklearn(max_samples: int | float | None) -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_fit_bookkeeping import forest_fit_bootstrap_sample_count

    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    clf = RandomForestClassifier(
        n_estimators=2,
        bootstrap=True,
        max_samples=max_samples,
        random_state=3,
    )
    clf.fit(X, y)

    assert forest_fit_bootstrap_sample_count(True, X.shape[0], max_samples) == clf._n_samples_bootstrap
    assert forest_fit_bootstrap_sample_count(False, X.shape[0], None) is None


def test_forest_fit_bootstrap_and_oob_preflight_match_sklearn_errors() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_fit_bookkeeping import (
        forest_fit_bootstrap_sample_count,
        forest_fit_require_bootstrap_for_oob,
    )

    X, y = make_classification(
        n_samples=25,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=11,
    )

    assert forest_fit_require_bootstrap_for_oob(True, False) is True

    with pytest.raises(ViolationError):
        forest_fit_bootstrap_sample_count(False, X.shape[0], 5)
    with pytest.raises(ValueError, match="bootstrap=False"):
        RandomForestClassifier(
            n_estimators=2,
            bootstrap=False,
            max_samples=5,
            random_state=0,
        ).fit(X, y)

    with pytest.raises(ViolationError):
        forest_fit_require_bootstrap_for_oob(False, True)
    with pytest.raises(ValueError, match="bootstrap=True"):
        RandomForestClassifier(
            n_estimators=2,
            bootstrap=False,
            oob_score=True,
            random_state=0,
        ).fit(X, y)


def test_forest_fit_additional_estimator_count_matches_warm_start_state() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_fit_bookkeeping import forest_fit_additional_estimator_count

    X, y = make_classification(
        n_samples=35,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        random_state=13,
    )
    clf = RandomForestClassifier(
        n_estimators=3,
        warm_start=True,
        random_state=5,
    )
    clf.fit(X, y)
    clf.n_estimators = 7

    assert forest_fit_additional_estimator_count(clf.n_estimators, len(clf.estimators_)) == 4


def test_forest_fit_oob_update_required_matches_fit_shell_cases() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_fit_bookkeeping import forest_fit_oob_update_required

    assert forest_fit_oob_update_required(False, 3, False) is False
    assert forest_fit_oob_update_required(True, 2, True) is True
    assert forest_fit_oob_update_required(True, 0, False) is True
    assert forest_fit_oob_update_required(True, 0, True) is False


def test_forest_fit_require_supported_oob_target_type_matches_sklearn_classifier_guard() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_fit_bookkeeping import (
        forest_fit_require_supported_oob_target_type,
    )

    X = np.arange(40, dtype=np.float64).reshape(10, 4)
    y = np.column_stack(
        [
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64),
            np.array([1, 2, 0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64),
        ]
    )

    target_type = type_of_target(y)
    with pytest.raises(ViolationError):
        forest_fit_require_supported_oob_target_type(target_type, True)
    with pytest.raises(ValueError, match="type of target cannot be used to compute OOB estimates"):
        RandomForestClassifier(
            n_estimators=2,
            bootstrap=True,
            oob_score=True,
            random_state=0,
        ).fit(X, y)

    assert forest_fit_require_supported_oob_target_type("continuous", False) is True


def test_contracts_reject_invalid_forest_fit_bookkeeping_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_fit_bookkeeping import (
        forest_fit_additional_estimator_count,
        forest_fit_bootstrap_sample_count,
        forest_fit_oob_update_required,
    )

    with pytest.raises(ViolationError):
        forest_fit_bootstrap_sample_count(True, 12, 1.5)

    with pytest.raises(ViolationError):
        forest_fit_additional_estimator_count(2, 3)

    with pytest.raises(ViolationError):
        forest_fit_oob_update_required(True, -1, False)  # type: ignore[arg-type]
