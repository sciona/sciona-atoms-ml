from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier


def test_semi_supervised_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.semi_supervised import (
        label_propagation_fit,
        label_propagation_predict,
        label_propagation_predict_proba,
        label_spreading_fit,
        label_spreading_predict,
        label_spreading_predict_proba,
        self_training_fit,
        self_training_predict,
        self_training_predict_proba,
        self_training_select_pseudo_labels,
    )

    assert callable(label_propagation_fit)
    assert callable(label_propagation_predict)
    assert callable(label_propagation_predict_proba)
    assert callable(label_spreading_fit)
    assert callable(label_spreading_predict)
    assert callable(label_spreading_predict_proba)
    assert callable(self_training_fit)
    assert callable(self_training_predict)
    assert callable(self_training_predict_proba)
    assert callable(self_training_select_pseudo_labels)


def test_label_propagation_rbf_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.semi_supervised import (
        label_propagation_fit,
        label_propagation_predict,
        label_propagation_predict_proba,
    )

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [8.0, 8.0], [8.0, 9.0], [9.0, 8.0]])
    y = np.array([0, -1, 0, 1, -1, 1])
    query = np.array([[0.0, 0.5], [8.5, 8.5]])

    state = label_propagation_fit(X, y, kernel="rbf", gamma=0.2, max_iter=100, tol=1e-4)
    expected = LabelPropagation(kernel="rbf", gamma=0.2, max_iter=100, tol=1e-4).fit(X, y)

    assert np.array_equal(state.classes.astype(int), expected.classes_)
    assert np.allclose(state.label_distributions, expected.label_distributions_)
    assert np.array_equal(state.transduction.astype(int), expected.transduction_)
    assert state.n_iter == expected.n_iter_
    assert np.allclose(label_propagation_predict_proba(query, state), expected.predict_proba(query))
    assert np.array_equal(label_propagation_predict(query, state).astype(int), expected.predict(query))


def test_label_spreading_knn_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.semi_supervised import (
        label_spreading_fit,
        label_spreading_predict,
        label_spreading_predict_proba,
    )

    X = np.array([[0.0, 0.0], [0.2, 0.1], [0.0, 1.0], [8.0, 8.0], [8.1, 8.4], [9.0, 8.0]])
    y = np.array([0, -1, 0, 1, -1, 1])
    query = np.array([[0.1, 0.2], [8.4, 8.2]])

    state = label_spreading_fit(X, y, kernel="knn", n_neighbors=2, alpha=0.3, max_iter=50, tol=1e-4)
    expected = LabelSpreading(kernel="knn", n_neighbors=2, alpha=0.3, max_iter=50, tol=1e-4).fit(X, y)

    assert np.array_equal(state.classes.astype(int), expected.classes_)
    assert np.allclose(state.label_distributions, expected.label_distributions_)
    assert np.array_equal(state.transduction.astype(int), expected.transduction_)
    assert state.n_iter == expected.n_iter_
    assert np.allclose(label_spreading_predict_proba(query, state), expected.predict_proba(query))
    assert np.array_equal(label_spreading_predict(query, state).astype(int), expected.predict(query))


def test_label_spreading_rbf_predict_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.semi_supervised import label_spreading_fit, label_spreading_predict

    X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    y = np.array([0, -1, 0, 1, -1, 1])
    query = np.array([[0.15], [5.15]])

    state = label_spreading_fit(X, y, kernel="rbf", gamma=1.0, alpha=0.2, max_iter=50, tol=1e-4)
    expected = LabelSpreading(kernel="rbf", gamma=1.0, alpha=0.2, max_iter=50, tol=1e-4).fit(X, y)

    assert np.array_equal(state.classes, expected.classes_)
    assert np.array_equal(label_spreading_predict(query, state), expected.predict(query))


def test_label_propagation_parameter_errors_match_contracts() -> None:
    from icontract.errors import ViolationError

    from sciona.atoms.ml.sklearn.semi_supervised import label_propagation_fit, label_spreading_fit

    X = np.array([[0.0], [1.0]])
    y = np.array([0, -1])

    with pytest.raises(ViolationError, match="n_neighbors must be positive"):
        label_propagation_fit(X, y, n_neighbors=0)
    with pytest.raises(ViolationError, match="alpha must be between zero and one"):
        label_spreading_fit(X, y, alpha=1.0)
    with pytest.raises(ViolationError, match="X and y must have equal sample count"):
        label_propagation_fit(X, np.array([0, -1, 1]))


def test_self_training_selection_matches_sklearn_rules() -> None:
    from sciona.atoms.ml.sklearn.semi_supervised import self_training_select_pseudo_labels

    max_proba = np.array([0.2, 0.76, 0.9, 0.75])
    assert np.array_equal(self_training_select_pseudo_labels(max_proba, threshold=0.75), np.array([False, True, True, False]))

    selected = self_training_select_pseudo_labels(max_proba, criterion="k_best", k_best=2)
    assert set(selected.tolist()) == {1, 2}

    all_selected = self_training_select_pseudo_labels(max_proba[:2], criterion="k_best", k_best=5)
    assert np.array_equal(all_selected, np.array([True, True]))


def test_self_training_fit_and_predict_match_sklearn_k_best() -> None:
    from sciona.atoms.ml.sklearn.semi_supervised import (
        self_training_fit,
        self_training_predict,
        self_training_predict_proba,
    )

    X = np.array([[0.0], [0.1], [0.2], [1.0], [5.0], [5.1], [5.2], [6.0]])
    y = np.array([0, -1, -1, 0, 1, -1, -1, 1])
    query = np.array([[0.15], [5.15]])
    estimator = KNeighborsClassifier(n_neighbors=1)

    state = self_training_fit(estimator, X, y, criterion="k_best", k_best=2, max_iter=3)
    expected = SelfTrainingClassifier(estimator, criterion="k_best", k_best=2, max_iter=3).fit(X, y)

    assert np.array_equal(state.transduction.astype(int), expected.transduction_)
    assert np.array_equal(state.labeled_iter, expected.labeled_iter_)
    assert state.n_iter == expected.n_iter_
    assert state.termination_condition == expected.termination_condition_
    assert np.array_equal(self_training_predict(query, state).astype(int), expected.predict(query))
    assert np.allclose(self_training_predict_proba(query, state), expected.predict_proba(query))


def test_self_training_max_iter_zero_matches_supervised_fit() -> None:
    from sciona.atoms.ml.sklearn.semi_supervised import self_training_fit, self_training_predict

    X = np.array([[0.0], [0.2], [5.0], [5.2], [9.0]])
    y = np.array([0, 0, 1, 1, -1])
    query = np.array([[0.1], [5.1]])
    estimator = KNeighborsClassifier(n_neighbors=1)

    state = self_training_fit(estimator, X, y, max_iter=0)
    expected = SelfTrainingClassifier(estimator, max_iter=0).fit(X, y)

    assert state.termination_condition == "max_iter"
    assert np.array_equal(state.transduction.astype(int), expected.transduction_)
    assert np.array_equal(self_training_predict(query, state).astype(int), expected.predict(query))
