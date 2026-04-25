from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.ensemble import RandomForestClassifier


def test_forest_classifier_targets_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_targets import (
        forest_classifier_class_weight_warning_required,
        forest_classifier_expanded_class_weight,
        forest_classifier_fit_targets,
        forest_classifier_validate_class_weight_preset,
    )

    assert callable(forest_classifier_fit_targets)
    assert callable(forest_classifier_validate_class_weight_preset)
    assert callable(forest_classifier_class_weight_warning_required)
    assert callable(forest_classifier_expanded_class_weight)


def test_forest_classifier_fit_targets_matches_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_targets import (
        forest_classifier_fit_targets,
    )

    y = np.array([0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 2, 1], dtype=np.int64).reshape(-1, 1)
    clf = RandomForestClassifier(n_estimators=2, random_state=0)
    clf.n_outputs_ = y.shape[1]

    encoded = clf._validate_y_class_weight(y)[0]
    state, actual_encoded = forest_classifier_fit_targets(y)

    assert len(state.classes) == 1
    assert np.array_equal(state.classes[0], clf.classes_[0].astype(object))
    assert state.n_classes == (int(clf.n_classes_[0]),)
    assert np.array_equal(actual_encoded, encoded.astype(np.int64))


def test_forest_classifier_fit_targets_matches_sklearn_multioutput() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_targets import (
        forest_classifier_fit_targets,
    )

    y = np.array(
        [[0, 1], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]],
        dtype=np.int64,
    )
    clf = RandomForestClassifier(n_estimators=2, random_state=0)
    clf.n_outputs_ = y.shape[1]

    encoded = clf._validate_y_class_weight(y)[0]
    state, actual_encoded = forest_classifier_fit_targets(y)

    assert [classes.tolist() for classes in state.classes] == [classes.tolist() for classes in clf.classes_]
    assert state.n_classes == tuple(int(value) for value in clf.n_classes_)
    assert np.array_equal(actual_encoded, encoded.astype(np.int64))


def test_forest_classifier_validate_class_weight_preset_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_targets import (
        forest_classifier_validate_class_weight_preset,
    )

    assert forest_classifier_validate_class_weight_preset(None) is None
    assert forest_classifier_validate_class_weight_preset("balanced") == "balanced"
    assert forest_classifier_validate_class_weight_preset("balanced_subsample") == "balanced_subsample"

    with pytest.raises(ValueError, match='Valid presets for class_weight include'):
        forest_classifier_validate_class_weight_preset("bad")


def test_forest_classifier_class_weight_warning_required_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_targets import (
        forest_classifier_class_weight_warning_required,
    )

    y = np.array([0, 1, 0, 1], dtype=np.int64).reshape(-1, 1)
    clf = RandomForestClassifier(
        n_estimators=2,
        random_state=0,
        class_weight="balanced",
        bootstrap=True,
        warm_start=True,
    )
    clf.n_outputs_ = 1

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf._validate_y_class_weight(y)

    assert forest_classifier_class_weight_warning_required("balanced", True) is True
    assert len(caught) == 1
    assert "not recommended for warm_start" in str(caught[0].message)
    assert forest_classifier_class_weight_warning_required(None, True) is False
    assert forest_classifier_class_weight_warning_required("balanced", False) is False


def test_forest_classifier_expanded_class_weight_matches_sklearn_single_output() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_targets import (
        forest_classifier_expanded_class_weight,
    )

    y = np.array([0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 2, 1], dtype=np.int64).reshape(-1, 1)

    for class_weight, bootstrap in [
        ("balanced", True),
        ("balanced_subsample", True),
        ("balanced_subsample", False),
        (None, True),
        ({0: 1.0, 1: 2.0, 2: 3.0}, True),
    ]:
        clf = RandomForestClassifier(
            n_estimators=2,
            random_state=0,
            class_weight=class_weight,
            bootstrap=bootstrap,
        )
        clf.n_outputs_ = y.shape[1]
        expected = clf._validate_y_class_weight(y)[1]

        result = forest_classifier_expanded_class_weight(y, class_weight, bootstrap=bootstrap)

        if expected is None:
            assert result is None
        else:
            assert np.allclose(result, expected.astype(np.float64))


def test_forest_classifier_expanded_class_weight_matches_sklearn_multioutput() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_targets import (
        forest_classifier_expanded_class_weight,
    )

    y = np.array(
        [[0, 1], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]],
        dtype=np.int64,
    )
    class_weight = [{0: 1.0, 1: 2.0, 2: 3.0}, {0: 1.0, 1: 4.0}]

    clf = RandomForestClassifier(
        n_estimators=2,
        random_state=0,
        class_weight=class_weight,
        bootstrap=True,
    )
    clf.n_outputs_ = y.shape[1]
    expected = clf._validate_y_class_weight(y)[1]

    result = forest_classifier_expanded_class_weight(y, class_weight, bootstrap=True)

    assert np.allclose(result, expected.astype(np.float64))


def test_contracts_reject_invalid_forest_classifier_target_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_classifier_targets import (
        forest_classifier_class_weight_warning_required,
        forest_classifier_expanded_class_weight,
        forest_classifier_fit_targets,
        forest_classifier_validate_class_weight_preset,
    )

    with pytest.raises(ViolationError):
        forest_classifier_fit_targets(np.array([0, 1, 0], dtype=np.int64))

    with pytest.raises(ViolationError):
        forest_classifier_validate_class_weight_preset({"a": 1.0})  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        forest_classifier_class_weight_warning_required({"a": 1.0}, True)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        forest_classifier_expanded_class_weight(
            np.array([0, 1, 0], dtype=np.int64),
            "balanced",
            bootstrap=True,
        )

    with pytest.raises(ViolationError):
        forest_classifier_expanded_class_weight(
            np.array([[0], [1], [0]], dtype=np.int64),
            [{"a": 1.0}, {"b": 2.0}],
            bootstrap=True,
        )
