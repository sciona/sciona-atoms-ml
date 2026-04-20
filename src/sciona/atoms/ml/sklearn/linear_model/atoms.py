"""Linear model atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .state_models import LinearRegressionState, RidgeClassifierState, RidgeCVState, RidgeState
from .witnesses import (
    witness_linear_regression_fit,
    witness_linear_regression_predict,
    witness_ridge_classifier_decision_function,
    witness_ridge_classifier_fit,
    witness_ridge_classifier_predict,
    witness_ridge_cv_fit,
    witness_ridge_cv_predict,
    witness_ridge_cv_scores,
    witness_ridge_fit,
    witness_ridge_predict,
    witness_ridge_regression,
)


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _target_1d_or_2d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim in {1, 2})


def _target_1d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim == 1)


def _same_sample_count(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(y).ndim in {1, 2} and np.asarray(X).shape[0] == np.asarray(y).shape[0])


def _sample_count_at_least_two(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[0] >= 2)


def _bool_value(value: bool) -> bool:
    return bool(isinstance(value, bool))


def _tol_valid(tol: float) -> bool:
    return bool(isinstance(tol, (int, float)) and not isinstance(tol, bool) and float(tol) >= 0.0 and np.isfinite(float(tol)))


def _n_jobs_valid(n_jobs: int | None) -> bool:
    return bool(n_jobs is None or (isinstance(n_jobs, int) and not isinstance(n_jobs, bool)))


def _sample_weight_valid(sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None, X: NDArray[np.float64]) -> bool:
    if sample_weight is None:
        return True
    values = np.atleast_1d(np.asarray(sample_weight, dtype=np.float64))
    return bool(
        values.ndim == 1
        and values.shape[0] in {1, np.asarray(X).shape[0]}
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
    )


def _finite_inputs(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(X, dtype=np.float64))) and np.all(np.isfinite(np.asarray(y, dtype=np.float64))))


def _finite_classifier_inputs(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(X, dtype=np.float64))) and np.all(np.isfinite(np.asarray(y, dtype=np.float64))))


def _alpha_valid(alpha: float | tuple[float, ...] | NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    n_outputs = 1 if np.asarray(y).ndim == 1 else np.asarray(y).shape[1]
    return bool(values.ndim == 1 and values.shape[0] in {1, n_outputs} and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _alphas_strictly_positive(alphas: float | tuple[float, ...] | NDArray[np.float64]) -> bool:
    try:
        values = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)) and np.all(values > 0.0))


def _solver_valid(solver: str) -> bool:
    return solver in {"auto", "cholesky"}


def _gcv_mode_valid(gcv_mode: str | None) -> bool:
    return gcv_mode in {None, "auto", "svd", "eigen"}


def _max_iter_valid(max_iter: int | None) -> bool:
    return bool(max_iter is None or (isinstance(max_iter, int) and not isinstance(max_iter, bool) and max_iter >= 1))


def _class_weight_valid(class_weight: dict[float, float] | str | None) -> bool:
    if class_weight is None or class_weight == "balanced":
        return True
    return bool(
        isinstance(class_weight, dict)
        and all(np.isfinite(float(key)) and np.isfinite(float(value)) and float(value) >= 0.0 for key, value in class_weight.items())
    )


def _state_valid(state: LinearRegressionState) -> bool:
    expected_coef_shape = (state.n_features_in,) if state.n_outputs == 1 else (state.n_outputs, state.n_features_in)
    return bool(
        state.coef.shape == expected_coef_shape
        and state.intercept.shape == (state.n_outputs,)
        and state.singular.ndim == 1
        and state.rank >= 0
        and state.n_features_in >= 1
        and state.n_outputs >= 1
        and isinstance(state.fit_intercept, bool)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.intercept))
        and np.all(np.isfinite(state.singular))
    )


def _ridge_state_valid(state: RidgeState) -> bool:
    expected_coef_shape = (state.n_features_in,) if state.n_outputs == 1 else (state.n_outputs, state.n_features_in)
    return bool(
        state.coef.shape == expected_coef_shape
        and state.intercept.shape == (state.n_outputs,)
        and state.alpha.ndim == 1
        and state.alpha.shape[0] in {1, state.n_outputs}
        and state.n_features_in >= 1
        and state.n_outputs >= 1
        and isinstance(state.fit_intercept, bool)
        and _solver_valid(state.solver)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.intercept))
        and np.all(np.isfinite(state.alpha))
        and np.all(state.alpha >= 0.0)
    )


def _ridge_cv_state_valid(state: RidgeCVState) -> bool:
    expected_coef_shape = (state.n_features_in,) if state.n_outputs == 1 else (state.n_outputs, state.n_features_in)
    return bool(
        state.coef.shape == expected_coef_shape
        and state.intercept.shape == (state.n_outputs,)
        and state.alpha.shape == (1,)
        and state.best_score.shape == (1,)
        and state.n_features_in >= 1
        and state.n_outputs >= 1
        and isinstance(state.fit_intercept, bool)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.intercept))
        and np.all(np.isfinite(state.alpha))
        and np.all(state.alpha > 0.0)
        and np.all(np.isfinite(state.best_score))
    )


def _ridge_classifier_state_valid(state: RidgeClassifierState) -> bool:
    n_classes = state.classes.shape[0]
    expected_rows = 1 if n_classes == 2 else n_classes
    return bool(
        state.coef.shape == (expected_rows, state.n_features_in)
        and state.intercept.shape == (expected_rows,)
        and state.classes.ndim == 1
        and n_classes >= 2
        and state.alpha.ndim == 1
        and state.alpha.shape[0] in {1, expected_rows}
        and state.n_features_in >= 1
        and isinstance(state.fit_intercept, bool)
        and _solver_valid(state.solver)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.intercept))
        and np.all(np.isfinite(state.classes))
        and np.all(np.diff(state.classes) > 0.0)
        and np.all(np.isfinite(state.alpha))
        and np.all(state.alpha >= 0.0)
    )


def _feature_count_matches(X: NDArray[np.float64], state: LinearRegressionState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _ridge_feature_count_matches(X: NDArray[np.float64], state: RidgeState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _ridge_cv_feature_count_matches(X: NDArray[np.float64], state: RidgeCVState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _ridge_classifier_feature_count_matches(X: NDArray[np.float64], state: RidgeClassifierState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: LinearRegressionState) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0],) if state.n_outputs == 1 else (np.asarray(X).shape[0], state.n_outputs)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _ridge_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RidgeState) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0],) if state.n_outputs == 1 else (np.asarray(X).shape[0], state.n_outputs)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _ridge_cv_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RidgeCVState) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0],) if state.n_outputs == 1 else (np.asarray(X).shape[0], state.n_outputs)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _ridge_coefficients_valid(result: NDArray[np.float64], X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    n_outputs = 1 if np.asarray(y).ndim == 1 else np.asarray(y).shape[1]
    expected_shape = (np.asarray(X).shape[1],) if n_outputs == 1 else (n_outputs, np.asarray(X).shape[1])
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _ridge_cv_scores_valid(result: NDArray[np.float64], alphas: float | tuple[float, ...] | NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    alpha_values = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    return bool(values.shape == (alpha_values.shape[0],) and np.all(np.isfinite(values)))


def _ridge_classifier_scores_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RidgeClassifierState) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0],) if state.classes.shape[0] == 2 else (np.asarray(X).shape[0], state.classes.shape[0])
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _ridge_classifier_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RidgeClassifierState) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isin(values, state.classes)))


def _center_and_rescale(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    fit_intercept: bool,
    sample_weight: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if fit_intercept:
        x_offset = np.average(X, axis=0, weights=sample_weight)
        y_offset = np.average(y, axis=0, weights=sample_weight)
        centered_x = X - x_offset
        centered_y = y - y_offset
    else:
        x_offset = np.zeros(X.shape[1], dtype=np.float64)
        y_offset = np.zeros(y.shape[1], dtype=np.float64)
        centered_x = X.copy()
        centered_y = y.copy()

    if sample_weight is not None:
        weight_sqrt = np.sqrt(sample_weight)
        centered_x = centered_x * weight_sqrt[:, np.newaxis]
        centered_y = centered_y * weight_sqrt[:, np.newaxis]
    return centered_x, centered_y, np.asarray(x_offset, dtype=np.float64), np.asarray(y_offset, dtype=np.float64)


def _center_without_rescale(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    fit_intercept: bool,
    sample_weight: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if fit_intercept:
        x_offset = np.average(X, axis=0, weights=sample_weight)
        y_offset = np.average(y, axis=0, weights=sample_weight)
        return X - x_offset, y - y_offset, np.asarray(x_offset, dtype=np.float64), np.asarray(y_offset, dtype=np.float64)
    return (
        X.copy(),
        y.copy(),
        np.zeros(X.shape[1], dtype=np.float64),
        np.zeros(y.shape[1], dtype=np.float64),
    )


def _expand_sample_weight(sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None, n_samples: int) -> NDArray[np.float64] | None:
    if sample_weight is None:
        return None
    weights = np.atleast_1d(np.asarray(sample_weight, dtype=np.float64))
    if weights.shape[0] == 1:
        return np.full(n_samples, weights[0], dtype=np.float64)
    return weights


def _ridge_solve_dense(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    solving_x = X
    solving_y = y
    if sample_weight is not None:
        weight_sqrt = np.sqrt(sample_weight)
        solving_x = solving_x * weight_sqrt[:, np.newaxis]
        solving_y = solving_y * weight_sqrt[:, np.newaxis]
    n_features = solving_x.shape[1]
    n_targets = solving_y.shape[1]
    alpha_values = alpha if alpha.shape[0] != 1 or n_targets == 1 else np.full(n_targets, alpha[0], dtype=np.float64)
    gram = solving_x.T @ solving_x
    rhs = solving_x.T @ solving_y
    if np.all(alpha_values == alpha_values[0]):
        system = gram.copy()
        system.flat[:: n_features + 1] += alpha_values[0]
        return np.asarray(linalg.solve(system, rhs, assume_a="pos", overwrite_a=False).T, dtype=np.float64)
    coefficients = np.empty((n_targets, n_features), dtype=np.float64)
    for output_index, current_alpha in enumerate(alpha_values):
        system = gram.copy()
        system.flat[:: n_features + 1] += current_alpha
        coefficients[output_index] = linalg.solve(system, rhs[:, output_index], assume_a="pos", overwrite_a=False).ravel()
    return coefficients


def _classifier_sample_weight(
    y: NDArray[np.float64],
    sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None,
    class_weight: dict[float, float] | str | None,
) -> NDArray[np.float64] | None:
    weights = _expand_sample_weight(sample_weight, y.shape[0])
    if class_weight is None:
        return weights
    if weights is None:
        weights = np.ones(y.shape[0], dtype=np.float64)
    if class_weight == "balanced":
        classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        class_weights = y.shape[0] / (classes.shape[0] * counts.astype(np.float64))
        return weights * class_weights[inverse]
    explicit = {float(key): float(value) for key, value in class_weight.items()}
    return weights * np.asarray([explicit.get(float(label), 1.0) for label in y], dtype=np.float64)


def _binarize_ridge_classes(y: NDArray[np.float64], classes: NDArray[np.float64]) -> NDArray[np.float64]:
    if classes.shape[0] == 2:
        return np.where(y == classes[1], 1.0, -1.0).reshape(-1, 1)
    encoded = np.full((y.shape[0], classes.shape[0]), -1.0, dtype=np.float64)
    for class_index, class_label in enumerate(classes):
        encoded[y == class_label, class_index] = 1.0
    return encoded


@register_atom(witness_linear_regression_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda copy_X: _bool_value(copy_X), "copy_X must be boolean")
@icontract.require(lambda tol: _tol_valid(tol), "tol must be non-negative")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be None or an integer")
@icontract.require(lambda positive: positive is False, "positive=True is outside this dense OLS atom scope")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must be non-negative and scalar or match sample count")
@icontract.ensure(lambda result: _state_valid(result), "linear regression state must contain finite fitted coefficients")
def linear_regression_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    fit_intercept: bool = True,
    copy_X: bool = True,
    tol: float = 1e-6,
    n_jobs: int | None = None,
    positive: bool = False,
    sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None = None,
) -> LinearRegressionState:
    """Fit dense ordinary least-squares coefficients."""
    del copy_X, tol, n_jobs, positive
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    y_was_1d = checked_y.ndim == 1
    if y_was_1d:
        checked_y = checked_y.reshape(-1, 1)
    weights = None if sample_weight is None else np.atleast_1d(np.asarray(sample_weight, dtype=np.float64))
    if weights is not None and weights.shape[0] == 1:
        weights = np.full(checked_x.shape[0], weights[0], dtype=np.float64)

    centered_x, centered_y, x_offset, y_offset = _center_and_rescale(checked_x, checked_y, fit_intercept, weights)
    cond = max(centered_x.shape) * np.finfo(centered_x.dtype).eps
    coefficients, _, rank, singular = linalg.lstsq(centered_x, centered_y, cond=cond)
    coefficients = coefficients.T
    intercept = y_offset - x_offset @ coefficients.T if fit_intercept else np.zeros(checked_y.shape[1], dtype=np.float64)
    if y_was_1d:
        coefficients = np.ravel(coefficients)
    return LinearRegressionState(
        coef=np.asarray(coefficients, dtype=np.float64),
        intercept=np.atleast_1d(np.asarray(intercept, dtype=np.float64)),
        rank=int(rank),
        singular=np.asarray(singular, dtype=np.float64),
        fit_intercept=fit_intercept,
        n_features_in=int(checked_x.shape[1]),
        n_outputs=int(checked_y.shape[1]),
    )


@register_atom(witness_linear_regression_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted linear regression state")
@icontract.require(lambda state: _state_valid(state), "state must be a fitted linear regression state")
@icontract.ensure(lambda result, X, state: _prediction_valid(result, X, state), "predictions must match fitted output width")
def linear_regression_predict(X: NDArray[np.float64], state: LinearRegressionState) -> NDArray[np.float64]:
    """Predict dense outputs from fitted ordinary least-squares coefficients."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    if state.n_outputs == 1:
        return np.asarray(checked_x @ state.coef + state.intercept[0], dtype=np.float64)
    return np.asarray(checked_x @ state.coef.T + state.intercept, dtype=np.float64)


@register_atom(witness_ridge_regression)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda alpha, y: _alpha_valid(alpha, y), "alpha must be non-negative and scalar or match output count")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must be non-negative and scalar or match sample count")
@icontract.require(lambda solver: _solver_valid(solver), "solver must be auto or cholesky")
@icontract.require(lambda max_iter: _max_iter_valid(max_iter), "max_iter must be None or positive")
@icontract.require(lambda tol: _tol_valid(tol), "tol must be non-negative")
@icontract.require(lambda positive: positive is False, "positive=True is outside this dense ridge atom scope")
@icontract.ensure(lambda result, X, y: _ridge_coefficients_valid(result, X, y), "ridge coefficients must match fitted output width")
def ridge_regression(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: float | tuple[float, ...] | NDArray[np.float64] = 1.0,
    *,
    sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None = None,
    solver: str = "auto",
    max_iter: int | None = None,
    tol: float = 1e-4,
    positive: bool = False,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Solve dense ridge-regression coefficients with a closed-form solver."""
    del solver, max_iter, tol, positive, random_state
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    y_was_1d = checked_y.ndim == 1
    if y_was_1d:
        checked_y = checked_y.reshape(-1, 1)
    weights = _expand_sample_weight(sample_weight, checked_x.shape[0])
    alpha_values = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    coefficients = _ridge_solve_dense(checked_x, checked_y, alpha_values, weights)
    if y_was_1d:
        return np.asarray(np.ravel(coefficients), dtype=np.float64)
    return np.asarray(coefficients, dtype=np.float64)


@register_atom(witness_ridge_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda alpha, y: _alpha_valid(alpha, y), "alpha must be non-negative and scalar or match output count")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda copy_X: _bool_value(copy_X), "copy_X must be boolean")
@icontract.require(lambda max_iter: _max_iter_valid(max_iter), "max_iter must be None or positive")
@icontract.require(lambda tol: _tol_valid(tol), "tol must be non-negative")
@icontract.require(lambda solver: _solver_valid(solver), "solver must be auto or cholesky")
@icontract.require(lambda positive: positive is False, "positive=True is outside this dense ridge atom scope")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must be non-negative and scalar or match sample count")
@icontract.ensure(lambda result: _ridge_state_valid(result), "ridge state must contain finite fitted coefficients")
def ridge_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    alpha: float | tuple[float, ...] | NDArray[np.float64] = 1.0,
    fit_intercept: bool = True,
    copy_X: bool = True,
    max_iter: int | None = None,
    tol: float = 1e-4,
    solver: str = "auto",
    positive: bool = False,
    random_state: int | None = None,
    sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None = None,
) -> RidgeState:
    """Fit dense ridge-regression coefficients and intercept."""
    del copy_X, max_iter, tol, positive, random_state
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    y_was_1d = checked_y.ndim == 1
    if y_was_1d:
        checked_y = checked_y.reshape(-1, 1)
    weights = _expand_sample_weight(sample_weight, checked_x.shape[0])
    centered_x, centered_y, x_offset, y_offset = _center_without_rescale(checked_x, checked_y, fit_intercept, weights)
    alpha_values = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    coefficients = _ridge_solve_dense(centered_x, centered_y, alpha_values, weights)
    intercept = y_offset - x_offset @ coefficients.T if fit_intercept else np.zeros(checked_y.shape[1], dtype=np.float64)
    if y_was_1d:
        coefficients = np.ravel(coefficients)
    return RidgeState(
        coef=np.asarray(coefficients, dtype=np.float64),
        intercept=np.atleast_1d(np.asarray(intercept, dtype=np.float64)),
        alpha=np.asarray(alpha_values, dtype=np.float64),
        fit_intercept=fit_intercept,
        solver="cholesky" if solver == "auto" else solver,
        n_features_in=int(checked_x.shape[1]),
        n_outputs=int(checked_y.shape[1]),
    )


@register_atom(witness_ridge_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _ridge_feature_count_matches(X, state), "X feature count must match fitted ridge state")
@icontract.require(lambda state: _ridge_state_valid(state), "state must be a fitted ridge state")
@icontract.ensure(lambda result, X, state: _ridge_prediction_valid(result, X, state), "predictions must match fitted output width")
def ridge_predict(X: NDArray[np.float64], state: RidgeState) -> NDArray[np.float64]:
    """Predict dense outputs from fitted ridge-regression coefficients."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    if state.n_outputs == 1:
        return np.asarray(checked_x @ state.coef + state.intercept[0], dtype=np.float64)
    return np.asarray(checked_x @ state.coef.T + state.intercept, dtype=np.float64)


@register_atom(witness_ridge_cv_scores)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X: _sample_count_at_least_two(X), "leave-one-out CV requires at least two samples")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda alphas: _alphas_strictly_positive(alphas), "alphas must be finite and strictly positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda scoring: scoring is None, "only default negative MSE scoring is covered")
@icontract.require(lambda cv: cv is None, "only leave-one-out CV is covered")
@icontract.require(lambda sample_weight: sample_weight is None, "sample_weight is outside this dense RidgeCV atom scope")
@icontract.ensure(lambda result, alphas: _ridge_cv_scores_valid(result, alphas), "CV scores must match the alpha grid")
def ridge_cv_scores(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    alphas: float | tuple[float, ...] | NDArray[np.float64] = (0.1, 1.0, 10.0),
    *,
    fit_intercept: bool = True,
    scoring: None = None,
    cv: None = None,
    sample_weight: None = None,
) -> NDArray[np.float64]:
    """Compute dense leave-one-out RidgeCV scores for each alpha."""
    del scoring, cv, sample_weight
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    y_was_1d = checked_y.ndim == 1
    checked_y_2d = checked_y.reshape(-1, 1) if y_was_1d else checked_y
    alpha_values = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    scores = np.empty(alpha_values.shape[0], dtype=np.float64)
    sample_indices = np.arange(checked_x.shape[0])
    for alpha_index, current_alpha in enumerate(alpha_values):
        predictions = np.empty_like(checked_y_2d, dtype=np.float64)
        for held_out in range(checked_x.shape[0]):
            train_mask = sample_indices != held_out
            train_y = checked_y_2d[train_mask, 0] if y_was_1d else checked_y_2d[train_mask]
            state = ridge_fit(
                checked_x[train_mask],
                train_y,
                alpha=float(current_alpha),
                fit_intercept=fit_intercept,
                solver="cholesky",
            )
            prediction = ridge_predict(checked_x[held_out : held_out + 1], state)
            predictions[held_out] = np.asarray(prediction, dtype=np.float64).reshape(1, -1)[0]
        scores[alpha_index] = -float(np.mean((checked_y_2d - predictions) ** 2))
    return scores


@register_atom(witness_ridge_cv_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X: _sample_count_at_least_two(X), "leave-one-out CV requires at least two samples")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda alphas: _alphas_strictly_positive(alphas), "alphas must be finite and strictly positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda scoring: scoring is None, "only default negative MSE scoring is covered")
@icontract.require(lambda cv: cv is None, "only leave-one-out CV is covered")
@icontract.require(lambda gcv_mode: _gcv_mode_valid(gcv_mode), "gcv_mode must be None, auto, svd, or eigen")
@icontract.require(lambda store_cv_results: store_cv_results is False, "store_cv_results is outside this atom scope")
@icontract.require(lambda alpha_per_target: alpha_per_target is False, "alpha_per_target is outside this atom scope")
@icontract.require(lambda sample_weight: sample_weight is None, "sample_weight is outside this dense RidgeCV atom scope")
@icontract.ensure(lambda result: _ridge_cv_state_valid(result), "RidgeCV state must contain finite fitted coefficients and selected alpha")
def ridge_cv_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    alphas: float | tuple[float, ...] | NDArray[np.float64] = (0.1, 1.0, 10.0),
    fit_intercept: bool = True,
    scoring: None = None,
    cv: None = None,
    gcv_mode: str | None = None,
    store_cv_results: bool = False,
    alpha_per_target: bool = False,
    sample_weight: None = None,
) -> RidgeCVState:
    """Fit dense ridge regression after leave-one-out alpha selection."""
    del scoring, cv, gcv_mode, store_cv_results, alpha_per_target, sample_weight
    alpha_values = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    scores = ridge_cv_scores(X, y, alpha_values, fit_intercept=fit_intercept)
    best_index = int(np.argmax(scores))
    selected_alpha = float(alpha_values[best_index])
    ridge_state = ridge_fit(X, y, alpha=selected_alpha, fit_intercept=fit_intercept, solver="cholesky")
    return RidgeCVState(
        coef=ridge_state.coef,
        intercept=ridge_state.intercept,
        alpha=np.asarray([selected_alpha], dtype=np.float64),
        best_score=np.asarray([scores[best_index]], dtype=np.float64),
        fit_intercept=fit_intercept,
        n_features_in=ridge_state.n_features_in,
        n_outputs=ridge_state.n_outputs,
    )


@register_atom(witness_ridge_cv_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _ridge_cv_feature_count_matches(X, state), "X feature count must match fitted RidgeCV state")
@icontract.require(lambda state: _ridge_cv_state_valid(state), "state must be a fitted RidgeCV state")
@icontract.ensure(lambda result, X, state: _ridge_cv_prediction_valid(result, X, state), "predictions must match fitted output width")
def ridge_cv_predict(X: NDArray[np.float64], state: RidgeCVState) -> NDArray[np.float64]:
    """Predict dense outputs from fitted RidgeCV coefficients."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    if state.n_outputs == 1:
        return np.asarray(checked_x @ state.coef + state.intercept[0], dtype=np.float64)
    return np.asarray(checked_x @ state.coef.T + state.intercept, dtype=np.float64)


@register_atom(witness_ridge_classifier_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_classifier_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda alpha, y: _alpha_valid(alpha, _binarize_ridge_classes(np.asarray(y, dtype=np.float64), np.unique(np.asarray(y, dtype=np.float64)))), "alpha must be non-negative and scalar or match output count")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda copy_X: _bool_value(copy_X), "copy_X must be boolean")
@icontract.require(lambda max_iter: _max_iter_valid(max_iter), "max_iter must be None or positive")
@icontract.require(lambda tol: _tol_valid(tol), "tol must be non-negative")
@icontract.require(lambda class_weight: _class_weight_valid(class_weight), "class_weight must be None, balanced, or a finite non-negative mapping")
@icontract.require(lambda solver: _solver_valid(solver), "solver must be auto or cholesky")
@icontract.require(lambda positive: positive is False, "positive=True is outside this dense ridge classifier atom scope")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must be non-negative and scalar or match sample count")
@icontract.ensure(lambda result: _ridge_classifier_state_valid(result), "ridge classifier state must contain finite fitted coefficients")
def ridge_classifier_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    alpha: float | tuple[float, ...] | NDArray[np.float64] = 1.0,
    fit_intercept: bool = True,
    copy_X: bool = True,
    max_iter: int | None = None,
    tol: float = 1e-4,
    class_weight: dict[float, float] | str | None = None,
    solver: str = "auto",
    positive: bool = False,
    random_state: int | None = None,
    sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None = None,
) -> RidgeClassifierState:
    """Fit dense ridge-classifier coefficients with numeric class labels."""
    del copy_X, max_iter, tol, positive, random_state
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    classes = np.unique(checked_y)
    if classes.shape[0] < 2:
        raise ValueError("ridge classifier requires at least two classes")
    encoded_y = _binarize_ridge_classes(checked_y, classes)
    weights = _classifier_sample_weight(checked_y, sample_weight, class_weight)
    centered_x, centered_y, x_offset, y_offset = _center_without_rescale(checked_x, encoded_y, fit_intercept, weights)
    alpha_values = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    coefficients = _ridge_solve_dense(centered_x, centered_y, alpha_values, weights)
    intercept = y_offset - x_offset @ coefficients.T if fit_intercept else np.zeros(encoded_y.shape[1], dtype=np.float64)
    return RidgeClassifierState(
        coef=np.asarray(coefficients, dtype=np.float64),
        intercept=np.atleast_1d(np.asarray(intercept, dtype=np.float64)),
        classes=np.asarray(classes, dtype=np.float64),
        alpha=np.asarray(alpha_values, dtype=np.float64),
        fit_intercept=fit_intercept,
        solver="cholesky" if solver == "auto" else solver,
        n_features_in=int(checked_x.shape[1]),
    )


@register_atom(witness_ridge_classifier_decision_function)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _ridge_classifier_feature_count_matches(X, state), "X feature count must match fitted ridge classifier state")
@icontract.require(lambda state: _ridge_classifier_state_valid(state), "state must be a fitted ridge classifier state")
@icontract.ensure(lambda result, X, state: _ridge_classifier_scores_valid(result, X, state), "decision scores must match fitted class width")
def ridge_classifier_decision_function(X: NDArray[np.float64], state: RidgeClassifierState) -> NDArray[np.float64]:
    """Compute dense ridge-classifier confidence scores."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    scores = checked_x @ state.coef.T + state.intercept
    if state.classes.shape[0] == 2:
        return np.asarray(scores.reshape(-1), dtype=np.float64)
    return np.asarray(scores, dtype=np.float64)


@register_atom(witness_ridge_classifier_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _ridge_classifier_feature_count_matches(X, state), "X feature count must match fitted ridge classifier state")
@icontract.require(lambda state: _ridge_classifier_state_valid(state), "state must be a fitted ridge classifier state")
@icontract.ensure(lambda result, X, state: _ridge_classifier_prediction_valid(result, X, state), "predictions must be fitted class labels")
def ridge_classifier_predict(X: NDArray[np.float64], state: RidgeClassifierState) -> NDArray[np.float64]:
    """Predict dense ridge-classifier labels."""
    scores = ridge_classifier_decision_function(X, state)
    if scores.ndim == 1:
        indices = (scores > 0.0).astype(np.int64)
    else:
        indices = np.argmax(scores, axis=1)
    return np.asarray(state.classes[indices], dtype=np.float64)
