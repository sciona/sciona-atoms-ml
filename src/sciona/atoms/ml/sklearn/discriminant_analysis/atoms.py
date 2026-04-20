"""Selected discriminant-analysis atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array
from sklearn.utils.validation import check_X_y

from sciona.ghost.registry import register_atom

from .state_models import QDAState
from .witnesses import (
    witness_qda_decision_function,
    witness_qda_fit,
    witness_qda_predict,
    witness_qda_predict_log_proba,
    witness_qda_predict_proba,
)


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _target_1d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim == 1)


def _sample_counts_match(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_y = np.asarray(y)
    return bool(values_x.ndim == 2 and values_y.ndim == 1 and values_x.shape[0] == values_y.shape[0])


def _finite_inputs(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X, dtype=np.float64)
    values_y = np.asarray(y, dtype=np.float64)
    return bool(np.all(np.isfinite(values_x)) and np.all(np.isfinite(values_y)))


def _at_least_two_classes(y: NDArray[np.float64]) -> bool:
    return bool(np.unique(np.asarray(y)).shape[0] >= 2)


def _class_counts_exceed_features(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_y = np.asarray(y)
    if values_x.ndim != 2 or values_y.ndim != 1:
        return False
    _, counts = np.unique(values_y, return_counts=True)
    return bool(np.all(counts > values_x.shape[1]))


def _priors_valid(priors: tuple[float, ...] | None, y: NDArray[np.float64]) -> bool:
    if priors is None:
        return True
    values = np.asarray(priors, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] == np.unique(y).shape[0] and np.all(values >= 0.0) and np.isclose(np.sum(values), 1.0))


def _state_valid(state: QDAState) -> bool:
    n_classes = state.classes.shape[0]
    return bool(
        n_classes >= 2
        and state.classes.ndim == 1
        and state.priors.shape == (n_classes,)
        and state.means.shape == (n_classes, state.n_features_in)
        and len(state.scalings) == n_classes
        and len(state.rotations) == n_classes
        and (state.covariance is None or len(state.covariance) == n_classes)
        and 0.0 <= state.reg_param <= 1.0
        and np.all(np.isfinite(state.classes))
        and np.all(np.isfinite(state.priors))
        and np.all(state.priors >= 0.0)
        and np.isclose(np.sum(state.priors), 1.0)
        and np.all(np.isfinite(state.means))
        and all(scaling.ndim == 1 and scaling.shape[0] == state.n_features_in for scaling in state.scalings)
        and all(rotation.shape == (state.n_features_in, state.n_features_in) for rotation in state.rotations)
        and all(np.all(np.isfinite(scaling)) and np.all(scaling > 0.0) for scaling in state.scalings)
        and all(np.all(np.isfinite(rotation)) for rotation in state.rotations)
        and (
            state.covariance is None
            or all(covariance.shape == (state.n_features_in, state.n_features_in) for covariance in state.covariance)
        )
        and (state.covariance is None or all(np.all(np.isfinite(covariance)) for covariance in state.covariance))
        and state.store_covariance == (state.covariance is not None)
    )


def _feature_count_matches(X: NDArray[np.float64], state: QDAState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _scores_valid(result: NDArray[np.float64], state: QDAState) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 2 and values.shape[1] == state.classes.shape[0] and np.all(np.isfinite(values)))


def _proba_valid(result: NDArray[np.float64], state: QDAState) -> bool:
    values = np.asarray(result)
    return bool(
        values.ndim == 2
        and values.shape[1] == state.classes.shape[0]
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(values.sum(axis=1), 1.0)
    )


def _prediction_valid(result: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 1 and np.all(np.isfinite(values)))


@register_atom(witness_qda_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must be finite")
@icontract.require(lambda y: _at_least_two_classes(y), "QDA requires at least two classes")
@icontract.require(lambda X, y: _class_counts_exceed_features(X, y), "each class must have more samples than features for the SVD solver")
@icontract.require(lambda priors, y: _priors_valid(priors, y), "priors must match classes and sum to one")
@icontract.require(lambda reg_param: 0.0 <= reg_param <= 1.0, "reg_param must lie in [0, 1]")
@icontract.require(lambda tol: tol >= 0.0, "tol must be non-negative")
@icontract.ensure(lambda result: _state_valid(result), "QDA state must contain finite class factors")
def qda_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    priors: tuple[float, ...] | None = None,
    reg_param: float = 0.0,
    store_covariance: bool = False,
    tol: float = 1e-4,
) -> QDAState:
    """Fit dense SVD-based Quadratic Discriminant Analysis state."""
    checked_x, checked_y = check_X_y(X, y, dtype=np.float64)
    classes = np.unique(checked_y)
    n_samples, n_features = checked_x.shape
    if priors is None:
        _, counts = np.unique(checked_y, return_counts=True)
        fitted_priors = counts.astype(np.float64) / float(n_samples)
    else:
        fitted_priors = np.asarray(priors, dtype=np.float64)

    means: list[NDArray[np.float64]] = []
    scalings: list[NDArray[np.float64]] = []
    rotations: list[NDArray[np.float64]] = []
    covariance: list[NDArray[np.float64]] = []
    for class_label in classes:
        class_x = checked_x[checked_y == class_label, :]
        mean = np.mean(class_x, axis=0)
        centered = class_x - mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        scaling = (singular_values**2) / (class_x.shape[0] - 1)
        scaling = ((1.0 - reg_param) * scaling) + reg_param
        rank = int(np.sum(scaling > tol))
        if rank < n_features:
            raise np.linalg.LinAlgError("The covariance matrix of at least one class is not full rank.")
        rotation = vt.T
        means.append(np.asarray(mean, dtype=np.float64).copy())
        scalings.append(np.asarray(scaling, dtype=np.float64).copy())
        rotations.append(np.asarray(rotation, dtype=np.float64).copy())
        if store_covariance:
            covariance.append(np.asarray((scaling * vt.T) @ vt, dtype=np.float64).copy())

    return QDAState(
        classes=np.asarray(classes, dtype=np.float64).copy(),
        priors=np.asarray(fitted_priors, dtype=np.float64).copy(),
        means=np.asarray(means, dtype=np.float64).copy(),
        scalings=tuple(scalings),
        rotations=tuple(rotations),
        covariance=tuple(covariance) if store_covariance else None,
        reg_param=float(reg_param),
        store_covariance=bool(store_covariance),
        n_features_in=int(n_features),
    )


@register_atom(witness_qda_decision_function)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted QDA state")
@icontract.require(lambda state: _state_valid(state), "state must be a fitted QDA state")
@icontract.ensure(lambda result, state: _scores_valid(result, state), "decision scores must be finite per-class values")
def qda_decision_function(
    X: NDArray[np.float64],
    state: QDAState,
) -> NDArray[np.float64]:
    """Compute QDA log posterior scores before probability normalization."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    norm2 = []
    for class_index in range(state.classes.shape[0]):
        rotation = state.rotations[class_index]
        scaling = state.scalings[class_index]
        centered = checked_x - state.means[class_index]
        transformed = np.dot(centered, rotation * (scaling ** (-0.5)))
        norm2.append(np.sum(transformed**2, axis=1))
    norm2_matrix = np.asarray(norm2, dtype=np.float64).T
    log_det = np.asarray([np.sum(np.log(scaling)) for scaling in state.scalings], dtype=np.float64)
    return -0.5 * (norm2_matrix + log_det) + np.log(state.priors)


@register_atom(witness_qda_predict_log_proba)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted QDA state")
@icontract.require(lambda state: _state_valid(state), "state must be a fitted QDA state")
@icontract.ensure(lambda result, state: _scores_valid(result, state), "log probabilities must be finite per-class values")
def qda_predict_log_proba(
    X: NDArray[np.float64],
    state: QDAState,
) -> NDArray[np.float64]:
    """Compute normalized QDA log probabilities for each class."""
    scores = qda_decision_function(X, state)
    log_likelihood = scores - scores.max(axis=1)[:, np.newaxis]
    return log_likelihood - np.log(np.exp(log_likelihood).sum(axis=1)[:, np.newaxis])


@register_atom(witness_qda_predict_proba)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted QDA state")
@icontract.require(lambda state: _state_valid(state), "state must be a fitted QDA state")
@icontract.ensure(lambda result, state: _proba_valid(result, state), "probabilities must be valid per-class rows")
def qda_predict_proba(
    X: NDArray[np.float64],
    state: QDAState,
) -> NDArray[np.float64]:
    """Compute normalized QDA class probabilities for each row."""
    return np.exp(qda_predict_log_proba(X, state))


@register_atom(witness_qda_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted QDA state")
@icontract.require(lambda state: _state_valid(state), "state must be a fitted QDA state")
@icontract.ensure(lambda result: _prediction_valid(result), "predictions must be finite class labels")
def qda_predict(
    X: NDArray[np.float64],
    state: QDAState,
) -> NDArray[np.float64]:
    """Predict QDA class labels from fitted state."""
    scores = qda_decision_function(X, state)
    return np.asarray(state.classes.take(scores.argmax(axis=1)), dtype=np.float64)
