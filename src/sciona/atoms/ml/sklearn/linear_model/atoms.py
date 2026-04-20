"""Linear model atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .state_models import LinearRegressionState, RidgeState
from .witnesses import (
    witness_linear_regression_fit,
    witness_linear_regression_predict,
    witness_ridge_fit,
    witness_ridge_predict,
    witness_ridge_regression,
)


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _target_1d_or_2d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim in {1, 2})


def _same_sample_count(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(y).ndim in {1, 2} and np.asarray(X).shape[0] == np.asarray(y).shape[0])


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


def _alpha_valid(alpha: float | tuple[float, ...] | NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    n_outputs = 1 if np.asarray(y).ndim == 1 else np.asarray(y).shape[1]
    return bool(values.ndim == 1 and values.shape[0] in {1, n_outputs} and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _solver_valid(solver: str) -> bool:
    return solver in {"auto", "cholesky"}


def _max_iter_valid(max_iter: int | None) -> bool:
    return bool(max_iter is None or (isinstance(max_iter, int) and not isinstance(max_iter, bool) and max_iter >= 1))


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


def _feature_count_matches(X: NDArray[np.float64], state: LinearRegressionState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _ridge_feature_count_matches(X: NDArray[np.float64], state: RidgeState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: LinearRegressionState) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0],) if state.n_outputs == 1 else (np.asarray(X).shape[0], state.n_outputs)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _ridge_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RidgeState) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0],) if state.n_outputs == 1 else (np.asarray(X).shape[0], state.n_outputs)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _ridge_coefficients_valid(result: NDArray[np.float64], X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    n_outputs = 1 if np.asarray(y).ndim == 1 else np.asarray(y).shape[1]
    expected_shape = (np.asarray(X).shape[1],) if n_outputs == 1 else (n_outputs, np.asarray(X).shape[1])
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


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
