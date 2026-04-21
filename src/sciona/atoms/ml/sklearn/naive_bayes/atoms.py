"""Dense Gaussian naive Bayes atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import GaussianNBState
from .witnesses import (
    witness_gaussian_nb_fit,
    witness_gaussian_nb_joint_log_likelihood,
    witness_gaussian_nb_predict,
    witness_gaussian_nb_predict_log_proba,
    witness_gaussian_nb_predict_proba,
    witness_gaussian_nb_update_mean_variance,
)


def _numeric_matrix(X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _nonconstant_matrix(X: NDArray[np.float64]) -> bool:
    values = np.asarray(X, dtype=np.float64)
    return bool(np.max(np.var(values, axis=0)) > 0.0)


def _int_label_vector(y: NDArray[np.int64], X: NDArray[np.float64]) -> bool:
    try:
        labels = np.asarray(y, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(labels.ndim == 1 and labels.shape[0] == np.asarray(X).shape[0] and np.unique(labels).shape[0] >= 2)


def _feature_count(X: NDArray[np.float64]) -> int:
    return int(np.asarray(X).shape[1])


def _row_count(X: NDArray[np.float64]) -> int:
    return int(np.asarray(X).shape[0])


def _optional_weights_valid(sample_weight: NDArray[np.float64] | None, n_rows: int) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and weights.shape[0] == n_rows and np.all(np.isfinite(weights)) and np.all(weights > 0.0))


def _optional_priors_valid(priors: NDArray[np.float64] | None, n_classes: int) -> bool:
    if priors is None:
        return True
    try:
        values = np.asarray(priors, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 1
        and values.shape[0] == n_classes
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
        and np.isclose(np.sum(values), 1.0)
    )


def _state_valid(state: GaussianNBState) -> bool:
    n_classes = int(state.classes.shape[0])
    return bool(
        state.classes.ndim == 1
        and n_classes >= 2
        and state.class_count.shape == (n_classes,)
        and state.class_prior.shape == (n_classes,)
        and state.theta.shape == (n_classes, state.n_features_in)
        and state.var.shape == (n_classes, state.n_features_in)
        and state.n_features_in >= 1
        and np.all(np.isfinite(state.class_count))
        and np.all(state.class_count > 0.0)
        and np.all(np.isfinite(state.class_prior))
        and np.all(state.class_prior > 0.0)
        and np.isclose(np.sum(state.class_prior), 1.0)
        and np.all(np.isfinite(state.theta))
        and np.all(np.isfinite(state.var))
        and np.all(state.var > 0.0)
        and np.isfinite(state.epsilon)
        and state.epsilon > 0.0
    )


def _update_inputs_valid(
    n_past: float,
    mu: NDArray[np.float64],
    var: NDArray[np.float64],
    X: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None,
) -> bool:
    means = np.asarray(mu, dtype=np.float64)
    variances = np.asarray(var, dtype=np.float64)
    values = np.asarray(X, dtype=np.float64)
    return bool(
        np.isfinite(n_past)
        and n_past >= 0.0
        and means.ndim == 1
        and variances.shape == means.shape
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(variances))
        and np.all(variances >= 0.0)
        and values.ndim == 2
        and values.shape[1] == means.shape[0]
        and np.all(np.isfinite(values))
        and _optional_weights_valid(sample_weight, values.shape[0])
    )


def _update_result_valid(
    result: tuple[NDArray[np.float64], NDArray[np.float64]],
    mu: NDArray[np.float64],
) -> bool:
    updated_mu, updated_var = result
    means = np.asarray(updated_mu, dtype=np.float64)
    variances = np.asarray(updated_var, dtype=np.float64)
    expected_shape = np.asarray(mu, dtype=np.float64).shape
    return bool(
        means.shape == expected_shape
        and variances.shape == expected_shape
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(variances))
        and np.all(variances >= 0.0)
    )


def _fit_result_valid(result: GaussianNBState, X: NDArray[np.float64], y: NDArray[np.int64]) -> bool:
    return bool(
        _state_valid(result)
        and result.n_features_in == _feature_count(X)
        and result.classes.shape[0] == np.unique(np.asarray(y, dtype=np.int64)).shape[0]
    )


def _matrix_against_state(X: NDArray[np.float64], state: GaussianNBState) -> bool:
    return bool(_numeric_matrix(X) and _state_valid(state) and _feature_count(X) == state.n_features_in)


def _class_matrix_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: GaussianNBState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (_row_count(X), state.classes.shape[0]) and np.all(np.isfinite(values)))


def _probability_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: GaussianNBState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (_row_count(X), state.classes.shape[0])
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _prediction_result_valid(result: NDArray[np.int64], X: NDArray[np.float64], state: GaussianNBState) -> bool:
    values = np.asarray(result, dtype=np.int64)
    return bool(values.shape == (_row_count(X),) and np.all(np.isin(values, state.classes)))


def _logsumexp(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    return np.squeeze(np.max(values, axis=axis, keepdims=True), axis=axis) + np.log(np.sum(np.exp(shifted), axis=axis))


@register_atom(witness_gaussian_nb_update_mean_variance)
@icontract.require(lambda n_past, mu, var, X, sample_weight: _update_inputs_valid(n_past, mu, var, X, sample_weight), "inputs must describe finite Gaussian feature statistics")
@icontract.ensure(lambda result, mu: _update_result_valid(result, mu), "updated mean and variance must match the feature shape")
def gaussian_nb_update_mean_variance(
    n_past: float,
    mu: NDArray[np.float64],
    var: NDArray[np.float64],
    X: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Combine existing Gaussian feature moments with a new dense sample block."""
    values = np.asarray(X, dtype=np.float64)
    means = np.asarray(mu, dtype=np.float64)
    variances = np.asarray(var, dtype=np.float64)
    if values.shape[0] == 0:
        return means.copy(), variances.copy()

    if sample_weight is None:
        n_new = float(values.shape[0])
        new_mu = np.mean(values, axis=0)
        new_var = np.var(values, axis=0)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        n_new = float(np.sum(weights))
        new_mu = np.average(values, axis=0, weights=weights)
        new_var = np.average((values - new_mu) ** 2, axis=0, weights=weights)

    if n_past == 0.0:
        return new_mu, new_var

    n_total = float(n_past + n_new)
    total_mu = (n_new * new_mu + n_past * means) / n_total
    old_ssd = n_past * variances
    new_ssd = n_new * new_var
    total_ssd = old_ssd + new_ssd + (n_new * n_past / n_total) * (means - new_mu) ** 2
    return total_mu, total_ssd / n_total


@register_atom(witness_gaussian_nb_fit)
@icontract.require(lambda X: _numeric_matrix(X), "X must be a dense finite numeric 2D matrix")
@icontract.require(lambda X: _nonconstant_matrix(X), "X must have at least one feature with positive variance")
@icontract.require(lambda X, y: _int_label_vector(y, X), "y must be an integer label vector with at least two classes")
@icontract.require(lambda X, sample_weight: _optional_weights_valid(sample_weight, _row_count(X)), "sample_weight must be positive and match X rows")
@icontract.require(lambda X, y, priors: _optional_priors_valid(priors, np.unique(np.asarray(y, dtype=np.int64)).shape[0]), "priors must be positive, sum to one, and match class count")
@icontract.require(lambda var_smoothing: np.isfinite(var_smoothing) and var_smoothing > 0.0, "var_smoothing must be positive")
@icontract.ensure(lambda result, X, y: _fit_result_valid(result, X, y), "state must contain one Gaussian per observed class")
def gaussian_nb_fit(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    priors: NDArray[np.float64] | None = None,
    var_smoothing: float = 1e-9,
    sample_weight: NDArray[np.float64] | None = None,
) -> GaussianNBState:
    """Fit dense Gaussian naive Bayes class means, variances, and priors."""
    values = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    classes = np.unique(labels).astype(np.int64)
    n_classes = int(classes.shape[0])
    n_features = int(values.shape[1])
    epsilon = float(var_smoothing * np.max(np.var(values, axis=0)))

    theta = np.zeros((n_classes, n_features), dtype=np.float64)
    variances = np.zeros((n_classes, n_features), dtype=np.float64)
    class_count = np.zeros(n_classes, dtype=np.float64)

    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    for idx, label in enumerate(classes):
        mask = labels == label
        class_values = values[mask]
        class_weights = None if weights is None else weights[mask]
        n_observed = float(class_values.shape[0]) if class_weights is None else float(np.sum(class_weights))
        new_theta, new_var = gaussian_nb_update_mean_variance(
            0.0,
            theta[idx],
            variances[idx],
            class_values,
            class_weights,
        )
        theta[idx] = new_theta
        variances[idx] = new_var
        class_count[idx] = n_observed

    variances = variances + epsilon
    if priors is None:
        class_prior = class_count / np.sum(class_count)
    else:
        class_prior = np.asarray(priors, dtype=np.float64)

    return GaussianNBState(
        classes=classes,
        class_count=class_count,
        class_prior=class_prior,
        theta=theta,
        var=variances,
        epsilon=epsilon,
        n_features_in=n_features,
    )


@register_atom(witness_gaussian_nb_joint_log_likelihood)
@icontract.require(lambda X, state: _matrix_against_state(X, state), "X must match a valid fitted GaussianNB state")
@icontract.ensure(lambda result, X, state: _class_matrix_result_valid(result, X, state), "joint log likelihood must have one column per class")
def gaussian_nb_joint_log_likelihood(X: NDArray[np.float64], state: GaussianNBState) -> NDArray[np.float64]:
    """Return unnormalized Gaussian naive Bayes log likelihoods by class."""
    values = np.asarray(X, dtype=np.float64)
    rows: list[NDArray[np.float64]] = []
    for idx in range(state.classes.shape[0]):
        joint = float(np.log(state.class_prior[idx]))
        normalizer = -0.5 * float(np.sum(np.log(2.0 * np.pi * state.var[idx])))
        squared = -0.5 * np.sum(((values - state.theta[idx]) ** 2) / state.var[idx], axis=1)
        rows.append(joint + normalizer + squared)
    return np.stack(rows, axis=1)


@register_atom(witness_gaussian_nb_predict_log_proba)
@icontract.require(lambda X, state: _matrix_against_state(X, state), "X must match a valid fitted GaussianNB state")
@icontract.ensure(lambda result, X, state: _class_matrix_result_valid(result, X, state), "log probabilities must have one column per class")
def gaussian_nb_predict_log_proba(X: NDArray[np.float64], state: GaussianNBState) -> NDArray[np.float64]:
    """Normalize Gaussian joint log likelihoods into class log probabilities."""
    joint = gaussian_nb_joint_log_likelihood(X, state)
    return joint - _logsumexp(joint, axis=1)[:, np.newaxis]


@register_atom(witness_gaussian_nb_predict_proba)
@icontract.require(lambda X, state: _matrix_against_state(X, state), "X must match a valid fitted GaussianNB state")
@icontract.ensure(lambda result, X, state: _probability_result_valid(result, X, state), "probabilities must be normalized by row")
def gaussian_nb_predict_proba(X: NDArray[np.float64], state: GaussianNBState) -> NDArray[np.float64]:
    """Return normalized Gaussian naive Bayes class probabilities."""
    return np.exp(gaussian_nb_predict_log_proba(X, state))


@register_atom(witness_gaussian_nb_predict)
@icontract.require(lambda X, state: _matrix_against_state(X, state), "X must match a valid fitted GaussianNB state")
@icontract.ensure(lambda result, X, state: _prediction_result_valid(result, X, state), "predictions must be fitted class labels")
def gaussian_nb_predict(X: NDArray[np.float64], state: GaussianNBState) -> NDArray[np.int64]:
    """Return the fitted class with largest Gaussian joint log likelihood."""
    joint = gaussian_nb_joint_log_likelihood(X, state)
    return state.classes[np.argmax(joint, axis=1)]
