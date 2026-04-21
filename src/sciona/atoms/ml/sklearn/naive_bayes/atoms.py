"""Dense Gaussian naive Bayes atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import ComplementNBState, GaussianNBState, MultinomialNBState
from .witnesses import (
    witness_complement_nb_count,
    witness_complement_nb_feature_log_prob,
    witness_complement_nb_fit,
    witness_complement_nb_joint_log_likelihood,
    witness_complement_nb_predict,
    witness_complement_nb_predict_log_proba,
    witness_complement_nb_predict_proba,
    witness_gaussian_nb_fit,
    witness_gaussian_nb_joint_log_likelihood,
    witness_gaussian_nb_predict,
    witness_gaussian_nb_predict_log_proba,
    witness_gaussian_nb_predict_proba,
    witness_gaussian_nb_update_mean_variance,
    witness_multinomial_nb_class_log_prior,
    witness_multinomial_nb_count,
    witness_multinomial_nb_feature_log_prob,
    witness_multinomial_nb_fit,
    witness_multinomial_nb_joint_log_likelihood,
    witness_multinomial_nb_predict,
    witness_multinomial_nb_predict_log_proba,
    witness_multinomial_nb_predict_proba,
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


def _nonnegative_matrix(X: NDArray[np.float64]) -> bool:
    return bool(_numeric_matrix(X) and np.all(np.asarray(X, dtype=np.float64) >= 0.0))


def _count_result_valid(
    result: tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
) -> bool:
    classes, class_count, feature_count = result
    n_classes = np.unique(np.asarray(y, dtype=np.int64)).shape[0]
    return bool(
        classes.shape == (n_classes,)
        and class_count.shape == (n_classes,)
        and feature_count.shape == (n_classes, _feature_count(X))
        and np.all(np.isfinite(class_count))
        and np.all(class_count > 0.0)
        and np.all(np.isfinite(feature_count))
        and np.all(feature_count >= 0.0)
    )


def _feature_count_matrix_valid(feature_count: NDArray[np.float64]) -> bool:
    values = np.asarray(feature_count, dtype=np.float64)
    return bool(values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] >= 1 and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _feature_log_prob_result_valid(result: NDArray[np.float64], feature_count: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == np.asarray(feature_count).shape
        and np.all(np.isfinite(values))
        and np.all(values <= 0.0)
        and np.allclose(np.sum(np.exp(values), axis=1), 1.0)
    )


def _class_count_valid(class_count: NDArray[np.float64]) -> bool:
    values = np.asarray(class_count, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] >= 2 and np.all(np.isfinite(values)) and np.all(values > 0.0))


def _class_prior_valid_for_count(class_prior: NDArray[np.float64] | None, n_classes: int) -> bool:
    return _optional_priors_valid(class_prior, n_classes)


def _class_log_prior_result_valid(result: NDArray[np.float64], class_count: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == np.asarray(class_count).shape
        and np.all(np.isfinite(values))
        and np.all(values <= 0.0)
        and np.isclose(np.sum(np.exp(values)), 1.0)
    )


def _multinomial_state_valid(state: MultinomialNBState) -> bool:
    n_classes = int(state.classes.shape[0])
    return bool(
        state.classes.ndim == 1
        and n_classes >= 2
        and state.class_count.shape == (n_classes,)
        and state.feature_count.shape == (n_classes, state.n_features_in)
        and state.class_log_prior.shape == (n_classes,)
        and state.feature_log_prob.shape == (n_classes, state.n_features_in)
        and state.n_features_in >= 1
        and np.all(np.isfinite(state.class_count))
        and np.all(state.class_count > 0.0)
        and np.all(np.isfinite(state.feature_count))
        and np.all(state.feature_count >= 0.0)
        and np.all(np.isfinite(state.class_log_prior))
        and np.isclose(np.sum(np.exp(state.class_log_prior)), 1.0)
        and np.all(np.isfinite(state.feature_log_prob))
        and np.allclose(np.sum(np.exp(state.feature_log_prob), axis=1), 1.0)
        and np.isfinite(state.alpha)
        and state.alpha > 0.0
    )


def _multinomial_fit_result_valid(result: MultinomialNBState, X: NDArray[np.float64], y: NDArray[np.int64]) -> bool:
    return bool(
        _multinomial_state_valid(result)
        and result.n_features_in == _feature_count(X)
        and result.classes.shape[0] == np.unique(np.asarray(y, dtype=np.int64)).shape[0]
    )


def _multinomial_matrix_against_state(X: NDArray[np.float64], state: MultinomialNBState) -> bool:
    return bool(_nonnegative_matrix(X) and _multinomial_state_valid(state) and _feature_count(X) == state.n_features_in)


def _multinomial_class_matrix_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: MultinomialNBState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (_row_count(X), state.classes.shape[0]) and np.all(np.isfinite(values)))


def _multinomial_probability_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: MultinomialNBState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (_row_count(X), state.classes.shape[0])
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _multinomial_prediction_result_valid(result: NDArray[np.int64], X: NDArray[np.float64], state: MultinomialNBState) -> bool:
    values = np.asarray(result, dtype=np.int64)
    return bool(values.shape == (_row_count(X),) and np.all(np.isin(values, state.classes)))


@register_atom(witness_multinomial_nb_count)
@icontract.require(lambda X: _nonnegative_matrix(X), "X must be a dense finite nonnegative 2D matrix")
@icontract.require(lambda X, y: _int_label_vector(y, X), "y must be an integer label vector with at least two classes")
@icontract.require(lambda X, sample_weight: _optional_weights_valid(sample_weight, _row_count(X)), "sample_weight must be positive and match X rows")
@icontract.ensure(lambda result, X, y: _count_result_valid(result, X, y), "counts must match class and feature dimensions")
def multinomial_nb_count(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate multinomial class counts and class-feature counts."""
    values = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    classes = np.unique(labels).astype(np.int64)
    label_matrix = (labels[:, np.newaxis] == classes[np.newaxis, :]).astype(np.float64)
    if sample_weight is not None:
        label_matrix *= np.asarray(sample_weight, dtype=np.float64)[:, np.newaxis]
    class_count = np.sum(label_matrix, axis=0)
    feature_count = label_matrix.T @ values
    return classes, class_count, feature_count


@register_atom(witness_multinomial_nb_feature_log_prob)
@icontract.require(lambda feature_count: _feature_count_matrix_valid(feature_count), "feature_count must be a finite nonnegative class-feature matrix")
@icontract.require(lambda alpha: np.isfinite(alpha) and alpha > 0.0, "alpha must be positive")
@icontract.ensure(lambda result, feature_count: _feature_log_prob_result_valid(result, feature_count), "feature log probabilities must normalize by class")
def multinomial_nb_feature_log_prob(feature_count: NDArray[np.float64], *, alpha: float = 1.0) -> NDArray[np.float64]:
    """Apply additive smoothing to multinomial feature counts."""
    smoothed = np.asarray(feature_count, dtype=np.float64) + float(alpha)
    smoothed_class_count = np.sum(smoothed, axis=1)
    return np.log(smoothed) - np.log(smoothed_class_count[:, np.newaxis])


@register_atom(witness_multinomial_nb_class_log_prior)
@icontract.require(lambda class_count: _class_count_valid(class_count), "class_count must be positive and one-dimensional")
@icontract.require(lambda class_count, class_prior: _class_prior_valid_for_count(class_prior, np.asarray(class_count).shape[0]), "class_prior must be positive, sum to one, and match class count")
@icontract.ensure(lambda result, class_count: _class_log_prior_result_valid(result, class_count), "class log priors must normalize across classes")
def multinomial_nb_class_log_prior(
    class_count: NDArray[np.float64],
    *,
    fit_prior: bool = True,
    class_prior: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute multinomial naive Bayes class log priors."""
    counts = np.asarray(class_count, dtype=np.float64)
    if class_prior is not None:
        return np.log(np.asarray(class_prior, dtype=np.float64))
    if fit_prior:
        return np.log(counts) - np.log(np.sum(counts))
    return np.full(counts.shape[0], -np.log(counts.shape[0]), dtype=np.float64)


@register_atom(witness_multinomial_nb_fit)
@icontract.require(lambda X: _nonnegative_matrix(X), "X must be a dense finite nonnegative 2D matrix")
@icontract.require(lambda X, y: _int_label_vector(y, X), "y must be an integer label vector with at least two classes")
@icontract.require(lambda alpha: np.isfinite(alpha) and alpha > 0.0, "alpha must be positive")
@icontract.require(lambda X, sample_weight: _optional_weights_valid(sample_weight, _row_count(X)), "sample_weight must be positive and match X rows")
@icontract.require(lambda X, y, class_prior: _optional_priors_valid(class_prior, np.unique(np.asarray(y, dtype=np.int64)).shape[0]), "class_prior must be positive, sum to one, and match class count")
@icontract.ensure(lambda result, X, y: _multinomial_fit_result_valid(result, X, y), "state must contain multinomial probabilities for each class")
def multinomial_nb_fit(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    alpha: float = 1.0,
    fit_prior: bool = True,
    class_prior: NDArray[np.float64] | None = None,
    sample_weight: NDArray[np.float64] | None = None,
) -> MultinomialNBState:
    """Fit dense multinomial naive Bayes counts and log probabilities."""
    classes, class_count, feature_count = multinomial_nb_count(X, y, sample_weight)
    feature_log_prob = multinomial_nb_feature_log_prob(feature_count, alpha=alpha)
    class_log_prior = multinomial_nb_class_log_prior(class_count, fit_prior=fit_prior, class_prior=class_prior)
    return MultinomialNBState(
        classes=classes,
        class_count=class_count,
        feature_count=feature_count,
        class_log_prior=class_log_prior,
        feature_log_prob=feature_log_prob,
        alpha=float(alpha),
        fit_prior=bool(fit_prior),
        n_features_in=_feature_count(X),
    )


@register_atom(witness_multinomial_nb_joint_log_likelihood)
@icontract.require(lambda X, state: _multinomial_matrix_against_state(X, state), "X must match a valid fitted MultinomialNB state")
@icontract.ensure(lambda result, X, state: _multinomial_class_matrix_result_valid(result, X, state), "joint log likelihood must have one column per class")
def multinomial_nb_joint_log_likelihood(X: NDArray[np.float64], state: MultinomialNBState) -> NDArray[np.float64]:
    """Return unnormalized multinomial naive Bayes log likelihoods by class."""
    return np.asarray(X, dtype=np.float64) @ state.feature_log_prob.T + state.class_log_prior


@register_atom(witness_multinomial_nb_predict_log_proba)
@icontract.require(lambda X, state: _multinomial_matrix_against_state(X, state), "X must match a valid fitted MultinomialNB state")
@icontract.ensure(lambda result, X, state: _multinomial_class_matrix_result_valid(result, X, state), "log probabilities must have one column per class")
def multinomial_nb_predict_log_proba(X: NDArray[np.float64], state: MultinomialNBState) -> NDArray[np.float64]:
    """Normalize multinomial joint log likelihoods into class log probabilities."""
    joint = multinomial_nb_joint_log_likelihood(X, state)
    return joint - _logsumexp(joint, axis=1)[:, np.newaxis]


@register_atom(witness_multinomial_nb_predict_proba)
@icontract.require(lambda X, state: _multinomial_matrix_against_state(X, state), "X must match a valid fitted MultinomialNB state")
@icontract.ensure(lambda result, X, state: _multinomial_probability_result_valid(result, X, state), "probabilities must be normalized by row")
def multinomial_nb_predict_proba(X: NDArray[np.float64], state: MultinomialNBState) -> NDArray[np.float64]:
    """Return normalized multinomial naive Bayes class probabilities."""
    return np.exp(multinomial_nb_predict_log_proba(X, state))


@register_atom(witness_multinomial_nb_predict)
@icontract.require(lambda X, state: _multinomial_matrix_against_state(X, state), "X must match a valid fitted MultinomialNB state")
@icontract.ensure(lambda result, X, state: _multinomial_prediction_result_valid(result, X, state), "predictions must be fitted class labels")
def multinomial_nb_predict(X: NDArray[np.float64], state: MultinomialNBState) -> NDArray[np.int64]:
    """Return the fitted class with largest multinomial joint log likelihood."""
    joint = multinomial_nb_joint_log_likelihood(X, state)
    return state.classes[np.argmax(joint, axis=1)]


def _complement_count_result_valid(
    result: tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
) -> bool:
    classes, class_count, feature_count, feature_all = result
    n_classes = np.unique(np.asarray(y, dtype=np.int64)).shape[0]
    return bool(
        classes.shape == (n_classes,)
        and class_count.shape == (n_classes,)
        and feature_count.shape == (n_classes, _feature_count(X))
        and feature_all.shape == (_feature_count(X),)
        and np.all(np.isfinite(class_count))
        and np.all(class_count > 0.0)
        and np.all(np.isfinite(feature_count))
        and np.all(feature_count >= 0.0)
        and np.all(np.isfinite(feature_all))
        and np.all(feature_all >= 0.0)
        and np.allclose(feature_all, np.sum(feature_count, axis=0))
    )


def _feature_all_valid(feature_count: NDArray[np.float64], feature_all: NDArray[np.float64]) -> bool:
    counts = np.asarray(feature_count, dtype=np.float64)
    totals = np.asarray(feature_all, dtype=np.float64)
    return bool(
        _feature_count_matrix_valid(counts)
        and totals.shape == (counts.shape[1],)
        and np.all(np.isfinite(totals))
        and np.all(totals >= 0.0)
        and np.allclose(totals, np.sum(counts, axis=0))
    )


def _complement_feature_log_prob_result_valid(result: NDArray[np.float64], feature_count: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == np.asarray(feature_count).shape and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _complement_state_valid(state: ComplementNBState) -> bool:
    n_classes = int(state.classes.shape[0])
    return bool(
        state.classes.ndim == 1
        and n_classes >= 2
        and state.class_count.shape == (n_classes,)
        and state.feature_count.shape == (n_classes, state.n_features_in)
        and state.feature_all.shape == (state.n_features_in,)
        and state.class_log_prior.shape == (n_classes,)
        and state.feature_log_prob.shape == (n_classes, state.n_features_in)
        and state.n_features_in >= 1
        and np.all(np.isfinite(state.class_count))
        and np.all(state.class_count > 0.0)
        and np.all(np.isfinite(state.feature_count))
        and np.all(state.feature_count >= 0.0)
        and np.allclose(state.feature_all, np.sum(state.feature_count, axis=0))
        and np.all(np.isfinite(state.class_log_prior))
        and np.isclose(np.sum(np.exp(state.class_log_prior)), 1.0)
        and np.all(np.isfinite(state.feature_log_prob))
        and np.all(state.feature_log_prob >= 0.0)
        and np.isfinite(state.alpha)
        and state.alpha > 0.0
    )


def _complement_fit_result_valid(result: ComplementNBState, X: NDArray[np.float64], y: NDArray[np.int64]) -> bool:
    return bool(
        _complement_state_valid(result)
        and result.n_features_in == _feature_count(X)
        and result.classes.shape[0] == np.unique(np.asarray(y, dtype=np.int64)).shape[0]
    )


def _complement_matrix_against_state(X: NDArray[np.float64], state: ComplementNBState) -> bool:
    return bool(_nonnegative_matrix(X) and _complement_state_valid(state) and _feature_count(X) == state.n_features_in)


def _complement_class_matrix_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: ComplementNBState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (_row_count(X), state.classes.shape[0]) and np.all(np.isfinite(values)))


def _complement_probability_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: ComplementNBState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (_row_count(X), state.classes.shape[0])
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _complement_prediction_result_valid(result: NDArray[np.int64], X: NDArray[np.float64], state: ComplementNBState) -> bool:
    values = np.asarray(result, dtype=np.int64)
    return bool(values.shape == (_row_count(X),) and np.all(np.isin(values, state.classes)))


@register_atom(witness_complement_nb_count)
@icontract.require(lambda X: _nonnegative_matrix(X), "X must be a dense finite nonnegative 2D matrix")
@icontract.require(lambda X, y: _int_label_vector(y, X), "y must be an integer label vector with at least two classes")
@icontract.require(lambda X, sample_weight: _optional_weights_valid(sample_weight, _row_count(X)), "sample_weight must be positive and match X rows")
@icontract.ensure(lambda result, X, y: _complement_count_result_valid(result, X, y), "counts must match class and feature dimensions")
def complement_nb_count(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate complement naive Bayes class, feature, and feature-total counts."""
    classes, class_count, feature_count = multinomial_nb_count(X, y, sample_weight)
    return classes, class_count, feature_count, np.sum(feature_count, axis=0)


@register_atom(witness_complement_nb_feature_log_prob)
@icontract.require(lambda feature_count, feature_all: _feature_all_valid(feature_count, feature_all), "feature_all must be the finite nonnegative sum of feature_count")
@icontract.require(lambda alpha: np.isfinite(alpha) and alpha > 0.0, "alpha must be positive")
@icontract.ensure(lambda result, feature_count: _complement_feature_log_prob_result_valid(result, feature_count), "complement feature weights must be finite and nonnegative")
def complement_nb_feature_log_prob(
    feature_count: NDArray[np.float64],
    feature_all: NDArray[np.float64],
    *,
    alpha: float = 1.0,
    norm: bool = False,
) -> NDArray[np.float64]:
    """Compute complement-smoothed feature weights for each class."""
    counts = np.asarray(feature_count, dtype=np.float64)
    totals = np.asarray(feature_all, dtype=np.float64)
    complement_count = totals + float(alpha) - counts
    logged = np.log(complement_count / np.sum(complement_count, axis=1, keepdims=True))
    if norm:
        return logged / np.sum(logged, axis=1, keepdims=True)
    return -logged


@register_atom(witness_complement_nb_fit)
@icontract.require(lambda X: _nonnegative_matrix(X), "X must be a dense finite nonnegative 2D matrix")
@icontract.require(lambda X, y: _int_label_vector(y, X), "y must be an integer label vector with at least two classes")
@icontract.require(lambda alpha: np.isfinite(alpha) and alpha > 0.0, "alpha must be positive")
@icontract.require(lambda X, sample_weight: _optional_weights_valid(sample_weight, _row_count(X)), "sample_weight must be positive and match X rows")
@icontract.require(lambda X, y, class_prior: _optional_priors_valid(class_prior, np.unique(np.asarray(y, dtype=np.int64)).shape[0]), "class_prior must be positive, sum to one, and match class count")
@icontract.ensure(lambda result, X, y: _complement_fit_result_valid(result, X, y), "state must contain complement weights for each class")
def complement_nb_fit(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    alpha: float = 1.0,
    fit_prior: bool = True,
    class_prior: NDArray[np.float64] | None = None,
    norm: bool = False,
    sample_weight: NDArray[np.float64] | None = None,
) -> ComplementNBState:
    """Fit dense complement naive Bayes counts and class weights."""
    classes, class_count, feature_count, feature_all = complement_nb_count(X, y, sample_weight)
    feature_log_prob = complement_nb_feature_log_prob(feature_count, feature_all, alpha=alpha, norm=norm)
    class_log_prior = multinomial_nb_class_log_prior(class_count, fit_prior=fit_prior, class_prior=class_prior)
    return ComplementNBState(
        classes=classes,
        class_count=class_count,
        feature_count=feature_count,
        feature_all=feature_all,
        class_log_prior=class_log_prior,
        feature_log_prob=feature_log_prob,
        alpha=float(alpha),
        fit_prior=bool(fit_prior),
        norm=bool(norm),
        n_features_in=_feature_count(X),
    )


@register_atom(witness_complement_nb_joint_log_likelihood)
@icontract.require(lambda X, state: _complement_matrix_against_state(X, state), "X must match a valid fitted ComplementNB state")
@icontract.ensure(lambda result, X, state: _complement_class_matrix_result_valid(result, X, state), "joint log likelihood must have one column per class")
def complement_nb_joint_log_likelihood(X: NDArray[np.float64], state: ComplementNBState) -> NDArray[np.float64]:
    """Return complement naive Bayes class scores."""
    return np.asarray(X, dtype=np.float64) @ state.feature_log_prob.T


@register_atom(witness_complement_nb_predict_log_proba)
@icontract.require(lambda X, state: _complement_matrix_against_state(X, state), "X must match a valid fitted ComplementNB state")
@icontract.ensure(lambda result, X, state: _complement_class_matrix_result_valid(result, X, state), "log probabilities must have one column per class")
def complement_nb_predict_log_proba(X: NDArray[np.float64], state: ComplementNBState) -> NDArray[np.float64]:
    """Normalize complement class scores into log probabilities."""
    joint = complement_nb_joint_log_likelihood(X, state)
    return joint - _logsumexp(joint, axis=1)[:, np.newaxis]


@register_atom(witness_complement_nb_predict_proba)
@icontract.require(lambda X, state: _complement_matrix_against_state(X, state), "X must match a valid fitted ComplementNB state")
@icontract.ensure(lambda result, X, state: _complement_probability_result_valid(result, X, state), "probabilities must be normalized by row")
def complement_nb_predict_proba(X: NDArray[np.float64], state: ComplementNBState) -> NDArray[np.float64]:
    """Return normalized complement naive Bayes class probabilities."""
    return np.exp(complement_nb_predict_log_proba(X, state))


@register_atom(witness_complement_nb_predict)
@icontract.require(lambda X, state: _complement_matrix_against_state(X, state), "X must match a valid fitted ComplementNB state")
@icontract.ensure(lambda result, X, state: _complement_prediction_result_valid(result, X, state), "predictions must be fitted class labels")
def complement_nb_predict(X: NDArray[np.float64], state: ComplementNBState) -> NDArray[np.int64]:
    """Return the fitted class with largest complement class score."""
    joint = complement_nb_joint_log_likelihood(X, state)
    return state.classes[np.argmax(joint, axis=1)]
