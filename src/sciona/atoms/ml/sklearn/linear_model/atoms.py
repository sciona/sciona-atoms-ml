"""Linear model atoms adapted from scikit-learn."""

from __future__ import annotations

from itertools import combinations
from math import log

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import binom
from sklearn.model_selection import KFold
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .state_models import (
    ARDRegressionState,
    BayesianRidgeState,
    LarsPathState,
    LarsState,
    LinearRegressionState,
    OrthogonalMatchingPursuitCVState,
    OrthogonalMatchingPursuitState,
    RidgeClassifierCVState,
    RidgeClassifierState,
    RidgeCVState,
    RidgeState,
    TheilSenRegressorState,
)
from .witnesses import (
    witness_ard_regression_fit,
    witness_ard_regression_predict,
    witness_ard_regression_predict_std,
    witness_bayesian_ridge_fit,
    witness_bayesian_ridge_predict,
    witness_bayesian_ridge_predict_std,
    witness_lars_fit,
    witness_lars_path,
    witness_lars_path_gram,
    witness_lars_predict,
    witness_linear_regression_fit,
    witness_linear_regression_predict,
    witness_orthogonal_matching_pursuit_fit,
    witness_orthogonal_matching_pursuit_cv_fit,
    witness_orthogonal_matching_pursuit_cv_predict,
    witness_orthogonal_matching_pursuit_predict,
    witness_omp_path_residues,
    witness_orthogonal_mp,
    witness_orthogonal_mp_gram,
    witness_ridge_classifier_cv_decision_function,
    witness_ridge_classifier_cv_fit,
    witness_ridge_classifier_cv_predict,
    witness_ridge_classifier_cv_scores,
    witness_ridge_classifier_decision_function,
    witness_ridge_classifier_fit,
    witness_ridge_classifier_predict,
    witness_ridge_cv_fit,
    witness_ridge_cv_predict,
    witness_ridge_cv_scores,
    witness_ridge_fit,
    witness_ridge_predict,
    witness_ridge_regression,
    witness_theil_sen_regressor_fit,
    witness_theil_sen_regressor_predict,
)


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _target_1d_or_2d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim in {1, 2})


def _target_1d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim == 1)


def _square_matrix(matrix: NDArray[np.float64]) -> bool:
    values = np.asarray(matrix)
    return bool(values.ndim == 2 and values.shape[0] == values.shape[1])


def _gram_and_xy_match(Gram: NDArray[np.float64], Xy: NDArray[np.float64]) -> bool:
    gram_values = np.asarray(Gram)
    xy_values = np.asarray(Xy)
    return bool(gram_values.ndim == 2 and xy_values.ndim in {1, 2} and xy_values.shape[0] == gram_values.shape[0])


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


def _finite_matrix(matrix: NDArray[np.float64]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(matrix, dtype=np.float64))))


def _finite_gram_inputs(Gram: NDArray[np.float64], Xy: NDArray[np.float64]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(Gram, dtype=np.float64))) and np.all(np.isfinite(np.asarray(Xy, dtype=np.float64))))


def _finite_classifier_inputs(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(X, dtype=np.float64))) and np.all(np.isfinite(np.asarray(y, dtype=np.float64))))


def _class_count_at_least_two(y: NDArray[np.float64]) -> bool:
    try:
        classes = np.unique(np.asarray(y, dtype=np.float64))
    except (TypeError, ValueError):
        return False
    return bool(classes.ndim == 1 and classes.shape[0] >= 2)


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


def _omp_n_nonzero_valid(n_nonzero_coefs: int | None, n_features: int) -> bool:
    return bool(
        n_nonzero_coefs is None
        or (
            isinstance(n_nonzero_coefs, int)
            and not isinstance(n_nonzero_coefs, bool)
            and 1 <= n_nonzero_coefs <= n_features
        )
    )


def _omp_tol_valid(tol: float | None) -> bool:
    return bool(tol is None or (isinstance(tol, (int, float)) and not isinstance(tol, bool) and np.isfinite(float(tol)) and float(tol) >= 0.0))


def _omp_precompute_valid(precompute: bool | str) -> bool:
    return bool(isinstance(precompute, bool) or precompute == "auto")


def _omp_cv_valid(cv: int | None, n_samples: int) -> bool:
    return bool(cv is None or (isinstance(cv, int) and not isinstance(cv, bool) and 2 <= cv <= n_samples))


def _omp_cv_max_iter_valid(max_iter: int | None, n_features: int) -> bool:
    return bool(max_iter is None or (isinstance(max_iter, int) and not isinstance(max_iter, bool) and 1 <= max_iter <= n_features))


def _omp_norms_valid(norms_squared: tuple[float, ...] | NDArray[np.float64] | None, Xy: NDArray[np.float64], tol: float | None) -> bool:
    if tol is None:
        return True
    if norms_squared is None:
        return False
    values = np.atleast_1d(np.asarray(norms_squared, dtype=np.float64))
    xy_values = np.asarray(Xy)
    n_targets = 1 if xy_values.ndim == 1 else xy_values.shape[1]
    return bool(values.ndim == 1 and values.shape[0] in {1, n_targets} and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _class_weight_valid(class_weight: dict[float, float] | str | None) -> bool:
    if class_weight is None or class_weight == "balanced":
        return True
    return bool(
        isinstance(class_weight, dict)
        and all(np.isfinite(float(key)) and np.isfinite(float(value)) and float(value) >= 0.0 for key, value in class_weight.items())
    )


def _positive_finite(value: float | None, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _nonnegative_finite(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0)


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


def _ridge_classifier_cv_state_valid(state: RidgeClassifierCVState) -> bool:
    n_classes = state.classes.shape[0]
    expected_rows = 1 if n_classes == 2 else n_classes
    return bool(
        state.coef.shape == (expected_rows, state.n_features_in)
        and state.intercept.shape == (expected_rows,)
        and state.classes.ndim == 1
        and n_classes >= 2
        and state.alpha.shape == (1,)
        and state.best_score.shape == (1,)
        and state.n_features_in >= 1
        and isinstance(state.fit_intercept, bool)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.intercept))
        and np.all(np.isfinite(state.classes))
        and np.all(np.diff(state.classes) > 0.0)
        and np.all(np.isfinite(state.alpha))
        and np.all(state.alpha > 0.0)
        and np.all(np.isfinite(state.best_score))
    )


def _omp_state_valid(state: OrthogonalMatchingPursuitState) -> bool:
    expected_coef_shape = (state.n_features_in,) if state.n_outputs == 1 else (state.n_outputs, state.n_features_in)
    return bool(
        state.coef.shape == expected_coef_shape
        and state.intercept.shape == (state.n_outputs,)
        and state.n_iter.shape == (state.n_outputs,)
        and state.n_features_in >= 1
        and state.n_outputs >= 1
        and _omp_n_nonzero_valid(state.n_nonzero_coefs, state.n_features_in)
        and _omp_tol_valid(state.tol)
        and isinstance(state.fit_intercept, bool)
        and _omp_precompute_valid(state.precompute)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.intercept))
        and np.all(state.n_iter >= 0)
        and np.all(state.n_iter <= state.n_features_in)
    )


def _omp_cv_state_valid(state: OrthogonalMatchingPursuitCVState) -> bool:
    return bool(
        state.coef.shape == (state.n_features_in,)
        and state.intercept.shape == (1,)
        and state.n_iter.shape == (1,)
        and state.n_features_in >= 2
        and 1 <= state.n_nonzero_coefs <= state.n_features_in
        and 1 <= state.max_iter <= state.n_features_in
        and state.n_nonzero_coefs <= state.max_iter
        and state.cv >= 2
        and isinstance(state.fit_intercept, bool)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.intercept))
        and np.all(state.n_iter >= 0)
        and np.all(state.n_iter <= state.n_features_in)
    )


def _feature_count_matches(X: NDArray[np.float64], state: LinearRegressionState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _ridge_feature_count_matches(X: NDArray[np.float64], state: RidgeState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _ridge_cv_feature_count_matches(X: NDArray[np.float64], state: RidgeCVState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _ridge_classifier_feature_count_matches(X: NDArray[np.float64], state: RidgeClassifierState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _ridge_classifier_cv_feature_count_matches(X: NDArray[np.float64], state: RidgeClassifierCVState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _omp_feature_count_matches(X: NDArray[np.float64], state: OrthogonalMatchingPursuitState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _omp_cv_feature_count_matches(X: NDArray[np.float64], state: OrthogonalMatchingPursuitCVState) -> bool:
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


def _ridge_classifier_cv_scores_output_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RidgeClassifierCVState) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0],) if state.classes.shape[0] == 2 else (np.asarray(X).shape[0], state.classes.shape[0])
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _ridge_classifier_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RidgeClassifierState) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isin(values, state.classes)))


def _ridge_classifier_cv_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RidgeClassifierCVState) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isin(values, state.classes)))


def _omp_coefficients_valid(result: NDArray[np.float64], X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    n_outputs = 1 if np.asarray(y).ndim == 1 else np.asarray(y).shape[1]
    expected_shape = (np.asarray(X).shape[1],) if n_outputs == 1 else (np.asarray(X).shape[1], n_outputs)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _omp_gram_coefficients_valid(result: NDArray[np.float64], Gram: NDArray[np.float64], Xy: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    n_outputs = 1 if np.asarray(Xy).ndim == 1 else np.asarray(Xy).shape[1]
    expected_shape = (np.asarray(Gram).shape[0],) if n_outputs == 1 else (np.asarray(Gram).shape[0], n_outputs)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _omp_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: OrthogonalMatchingPursuitState) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0],) if state.n_outputs == 1 else (np.asarray(X).shape[0], state.n_outputs)
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))


def _omp_cv_residues_valid(result: NDArray[np.float64], X_test: NDArray[np.float64], max_iter: int) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 2 and 1 <= values.shape[0] <= max_iter and values.shape[1] == np.asarray(X_test).shape[0] and np.all(np.isfinite(values)))


def _omp_cv_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isfinite(values)))


def _bayesian_ridge_state_valid(state: BayesianRidgeState) -> bool:
    return bool(
        state.coef.shape == (state.n_features_in,)
        and state.sigma.shape == (state.n_features_in, state.n_features_in)
        and state.scores.ndim == 1
        and state.x_offset.shape == (state.n_features_in,)
        and state.x_scale.shape == (state.n_features_in,)
        and state.n_iter >= 1
        and state.alpha > 0.0
        and state.lambda_ > 0.0
        and isinstance(state.fit_intercept, bool)
        and isinstance(state.compute_score, bool)
        and np.isfinite(state.intercept)
        and np.isfinite(state.alpha)
        and np.isfinite(state.lambda_)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.sigma))
        and np.allclose(state.sigma, state.sigma.T)
        and np.all(np.isfinite(state.scores))
        and np.all(np.isfinite(state.x_offset))
        and np.all(np.isfinite(state.x_scale))
    )


def _bayesian_ridge_feature_count_matches(X: NDArray[np.float64], state: BayesianRidgeState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _bayesian_ridge_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isfinite(values)))


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


@register_atom(witness_bayesian_ridge_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X: _sample_count_at_least_two(X), "X must contain at least two samples")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda max_iter: isinstance(max_iter, int) and not isinstance(max_iter, bool) and max_iter >= 1, "max_iter must be positive")
@icontract.require(lambda tol: _positive_finite(tol), "tol must be positive")
@icontract.require(lambda alpha_1: _nonnegative_finite(alpha_1), "alpha_1 must be non-negative")
@icontract.require(lambda alpha_2: _nonnegative_finite(alpha_2), "alpha_2 must be non-negative")
@icontract.require(lambda lambda_1: _nonnegative_finite(lambda_1), "lambda_1 must be non-negative")
@icontract.require(lambda lambda_2: _nonnegative_finite(lambda_2), "lambda_2 must be non-negative")
@icontract.require(lambda alpha_init: _positive_finite(alpha_init, allow_none=True), "alpha_init must be positive when provided")
@icontract.require(lambda lambda_init: _positive_finite(lambda_init, allow_none=True), "lambda_init must be positive when provided")
@icontract.require(lambda compute_score: _bool_value(compute_score), "compute_score must be boolean")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda copy_X: _bool_value(copy_X), "copy_X must be boolean")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must be non-negative and scalar or match sample count")
@icontract.ensure(lambda result: _bayesian_ridge_state_valid(result), "Bayesian ridge state must contain finite posterior parameters")
def bayesian_ridge_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    alpha_init: float | None = None,
    lambda_init: float | None = None,
    compute_score: bool = False,
    fit_intercept: bool = True,
    copy_X: bool = True,
    sample_weight: float | tuple[float, ...] | NDArray[np.float64] | None = None,
) -> BayesianRidgeState:
    """Fit dense Bayesian ridge posterior parameters."""
    del copy_X
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    weights = _expand_sample_weight(sample_weight, checked_x.shape[0])
    n_samples, n_features = checked_x.shape
    sw_sum = float(n_samples) if weights is None else float(np.sum(weights))
    if weights is None:
        y_var = float(np.var(checked_y))
    else:
        y_mean = float(np.average(checked_y, weights=weights))
        y_var = float(np.average((checked_y - y_mean) ** 2, weights=weights))

    centered_x_2d, centered_y_2d, x_offset, y_offset = _center_and_rescale(
        checked_x,
        checked_y.reshape(-1, 1),
        fit_intercept,
        weights,
    )
    centered_y = np.ravel(centered_y_2d)
    eps = np.finfo(np.float64).eps
    alpha = float(alpha_init) if alpha_init is not None else float(1.0 / (y_var + eps))
    lambda_value = float(lambda_init) if lambda_init is not None else 1.0

    scores: list[float] = []
    coef_old: NDArray[np.float64] | None = None
    XT_y = np.dot(centered_x_2d.T, centered_y)
    U, singular_values, Vh_full = linalg.svd(centered_x_2d, full_matrices=(n_samples < n_features))
    k_rank = len(singular_values)
    eigen_vals = singular_values**2
    eigen_vals_full = np.zeros(n_features, dtype=np.float64)
    eigen_vals_full[:k_rank] = eigen_vals
    Vh = Vh_full[:k_rank, :]
    coef = np.zeros(n_features, dtype=np.float64)
    sse = 0.0

    iteration = 0
    for iteration in range(max_iter):
        coef, sse = _bayesian_ridge_update_coef(centered_x_2d, centered_y, XT_y, U, Vh, eigen_vals, alpha, lambda_value)
        if compute_score:
            scores.append(
                _bayesian_ridge_log_marginal_likelihood(
                    n_samples=n_samples,
                    n_features=n_features,
                    sw_sum=sw_sum,
                    eigen_vals=eigen_vals,
                    alpha=alpha,
                    lambda_=lambda_value,
                    coef=coef,
                    sse=sse,
                    alpha_1=alpha_1,
                    alpha_2=alpha_2,
                    lambda_1=lambda_1,
                    lambda_2=lambda_2,
                )
            )
        gamma = float(np.sum((alpha * eigen_vals) / (lambda_value + alpha * eigen_vals)))
        lambda_value = float((gamma + 2.0 * lambda_1) / (np.sum(coef**2) + 2.0 * lambda_2))
        alpha = float((sw_sum - gamma + 2.0 * alpha_1) / (sse + 2.0 * alpha_2))
        if iteration != 0 and coef_old is not None and np.sum(np.abs(coef_old - coef)) < tol:
            break
        coef_old = np.copy(coef)

    n_iter = int(iteration + 1)
    final_coef, final_sse = _bayesian_ridge_update_coef(centered_x_2d, centered_y, XT_y, U, Vh, eigen_vals, alpha, lambda_value)
    if compute_score:
        scores.append(
            _bayesian_ridge_log_marginal_likelihood(
                n_samples=n_samples,
                n_features=n_features,
                sw_sum=sw_sum,
                eigen_vals=eigen_vals,
                alpha=alpha,
                lambda_=lambda_value,
                coef=coef,
                sse=final_sse,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
            )
        )
    sigma = np.dot(Vh_full.T, Vh_full / (alpha * eigen_vals_full + lambda_value)[:, np.newaxis])
    intercept = float(y_offset[0] - np.dot(x_offset, final_coef)) if fit_intercept else 0.0
    return BayesianRidgeState(
        coef=np.asarray(final_coef, dtype=np.float64),
        intercept=intercept,
        alpha=float(alpha),
        lambda_=float(lambda_value),
        sigma=np.asarray(sigma, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        n_iter=n_iter,
        x_offset=np.asarray(x_offset, dtype=np.float64),
        x_scale=np.ones(n_features, dtype=np.float64),
        fit_intercept=fit_intercept,
        compute_score=compute_score,
        n_features_in=int(n_features),
    )


@register_atom(witness_bayesian_ridge_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _bayesian_ridge_feature_count_matches(X, state), "X feature count must match fitted Bayesian ridge state")
@icontract.require(lambda state: _bayesian_ridge_state_valid(state), "state must be a fitted Bayesian ridge state")
@icontract.require(lambda return_std: return_std is False, "use bayesian_ridge_predict_std for posterior standard deviations")
@icontract.ensure(lambda result, X: _bayesian_ridge_prediction_valid(result, X), "predictions must be finite per-row values")
def bayesian_ridge_predict(
    X: NDArray[np.float64],
    state: BayesianRidgeState,
    *,
    return_std: bool = False,
) -> NDArray[np.float64]:
    """Predict posterior mean values from fitted Bayesian ridge state."""
    del return_std
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    return np.asarray(np.dot(checked_x, state.coef) + state.intercept, dtype=np.float64)


@register_atom(witness_bayesian_ridge_predict_std)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _bayesian_ridge_feature_count_matches(X, state), "X feature count must match fitted Bayesian ridge state")
@icontract.require(lambda state: _bayesian_ridge_state_valid(state), "state must be a fitted Bayesian ridge state")
@icontract.ensure(lambda result, X: _bayesian_ridge_prediction_valid(result, X), "standard deviations must be finite per-row values")
def bayesian_ridge_predict_std(X: NDArray[np.float64], state: BayesianRidgeState) -> NDArray[np.float64]:
    """Predict posterior standard deviations from fitted Bayesian ridge state."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    sigmas_squared = (np.dot(checked_x, state.sigma) * checked_x).sum(axis=1)
    return np.asarray(np.sqrt(sigmas_squared + (1.0 / state.alpha)), dtype=np.float64)


def _bayesian_ridge_update_coef(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    XT_y: NDArray[np.float64],
    U: NDArray[np.float64],
    Vh: NDArray[np.float64],
    eigen_vals: NDArray[np.float64],
    alpha: float,
    lambda_: float,
) -> tuple[NDArray[np.float64], float]:
    n_samples, n_features = X.shape
    if n_samples > n_features:
        coef = np.linalg.multi_dot([Vh.T, Vh / (eigen_vals + lambda_ / alpha)[:, np.newaxis], XT_y])
    else:
        coef = np.linalg.multi_dot([X.T, U / (eigen_vals + lambda_ / alpha)[None, :], U.T, y])
    sse = float(np.sum((y - np.dot(X, coef)) ** 2))
    return np.asarray(coef, dtype=np.float64), sse


def _bayesian_ridge_log_marginal_likelihood(
    *,
    n_samples: int,
    n_features: int,
    sw_sum: float,
    eigen_vals: NDArray[np.float64],
    alpha: float,
    lambda_: float,
    coef: NDArray[np.float64],
    sse: float,
    alpha_1: float,
    alpha_2: float,
    lambda_1: float,
    lambda_2: float,
) -> float:
    if n_samples > n_features:
        logdet_sigma = -np.sum(np.log(lambda_ + alpha * eigen_vals))
    else:
        logdet_values = np.full(n_features, lambda_, dtype=np.float64)
        logdet_values[:n_samples] += alpha * eigen_vals
        logdet_sigma = -np.sum(np.log(logdet_values))
    score = lambda_1 * log(lambda_) - lambda_2 * lambda_
    score += alpha_1 * log(alpha) - alpha_2 * alpha
    score += 0.5 * (
        n_features * log(lambda_)
        + sw_sum * log(alpha)
        - alpha * sse
        - lambda_ * np.sum(coef**2)
        + logdet_sigma
        - sw_sum * log(2.0 * np.pi)
    )
    return float(score)


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


def _resolve_omp_n_nonzero(n_nonzero_coefs: int | None, tol: float | None, n_features: int, *, gram_default: bool = False) -> int:
    if n_nonzero_coefs is not None:
        return int(n_nonzero_coefs)
    if tol is not None:
        return n_features
    if gram_default:
        return int(0.1 * n_features)
    return max(int(0.1 * n_features), 1)


def _omp_solve_single(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    n_nonzero_coefs: int,
    tol: float | None,
) -> tuple[NDArray[np.float64], int]:
    residual = y.copy()
    active: list[int] = []
    sign_active: list[float] = []
    coefficients = np.zeros(X.shape[1], dtype=np.float64)
    max_features = X.shape[1] if tol is not None else n_nonzero_coefs
    for _ in range(max_features):
        correlations = X.T @ residual
        if active:
            correlations[np.asarray(active, dtype=np.int64)] = 0.0
        selected = int(np.argmax(np.abs(correlations)))
        if abs(correlations[selected]) <= np.finfo(np.float64).eps:
            break
        active.append(selected)
        active_x = X[:, active]
        if np.linalg.matrix_rank(active_x) < len(active):
            active.pop()
            break
        gamma, _, _, _ = linalg.lstsq(active_x, y, cond=None)
        residual = y - active_x @ gamma
        if tol is not None and float(residual @ residual) <= tol:
            break
    if active:
        coefficients[np.asarray(active, dtype=np.int64)] = np.asarray(gamma, dtype=np.float64)
    return coefficients, len(active)


def _omp_path_coefficients_single(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    max_iter: int,
) -> tuple[NDArray[np.float64], int]:
    residual = y.copy()
    active: list[int] = []
    path = np.zeros((X.shape[1], max_iter), dtype=np.float64)
    for step in range(max_iter):
        correlations = X.T @ residual
        if active:
            correlations[np.asarray(active, dtype=np.int64)] = 0.0
        selected = int(np.argmax(np.abs(correlations)))
        if abs(correlations[selected]) <= np.finfo(np.float64).eps:
            return path[:, :step], step
        active.append(selected)
        active_x = X[:, active]
        if np.linalg.matrix_rank(active_x) < len(active):
            active.pop()
            return path[:, :step], step
        gamma, _, _, _ = linalg.lstsq(active_x, y, cond=None)
        residual = y - active_x @ gamma
        path[np.asarray(active, dtype=np.int64), step] = np.asarray(gamma, dtype=np.float64)
    return path, max_iter


def _omp_solve(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    n_nonzero_coefs: int,
    tol: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    y_2d = y.reshape(-1, 1) if y.ndim == 1 else y
    coefficients = np.zeros((X.shape[1], y_2d.shape[1]), dtype=np.float64)
    n_iters = np.zeros(y_2d.shape[1], dtype=np.int64)
    for target_index in range(y_2d.shape[1]):
        coef, n_iter = _omp_solve_single(X, y_2d[:, target_index], n_nonzero_coefs, tol)
        coefficients[:, target_index] = coef
        n_iters[target_index] = n_iter
    if y.ndim == 1:
        return coefficients[:, 0], n_iters
    return coefficients, n_iters


def _gram_omp_solve_single(
    Gram: NDArray[np.float64],
    Xy: NDArray[np.float64],
    n_nonzero_coefs: int,
    tol_0: float | None,
    tol: float | None,
) -> tuple[NDArray[np.float64], int]:
    active: list[int] = []
    coefficients = np.zeros(Gram.shape[0], dtype=np.float64)
    alpha = Xy.copy()
    tol_curr = tol_0
    previous_delta = 0.0
    max_features = Gram.shape[0] if tol is not None else n_nonzero_coefs
    for _ in range(max_features):
        if active:
            alpha[np.asarray(active, dtype=np.int64)] = 0.0
        selected = int(np.argmax(np.abs(alpha)))
        if abs(alpha[selected]) <= np.finfo(np.float64).eps:
            break
        active.append(selected)
        active_index = np.asarray(active, dtype=np.int64)
        active_gram = Gram[np.ix_(active_index, active_index)]
        active_xy = Xy[active_index]
        try:
            gamma = linalg.solve(active_gram, active_xy, assume_a="sym")
        except linalg.LinAlgError:
            gamma, _, _, _ = linalg.lstsq(active_gram, active_xy, cond=None)
        beta = Gram[:, active_index] @ gamma
        alpha = Xy - beta
        if tol is not None and tol_curr is not None:
            tol_curr += previous_delta
            previous_delta = float(gamma @ beta[active_index])
            tol_curr -= previous_delta
            if abs(tol_curr) <= tol:
                break
    if active:
        coefficients[np.asarray(active, dtype=np.int64)] = np.asarray(gamma, dtype=np.float64)
    return coefficients, len(active)


def _gram_omp_solve(
    Gram: NDArray[np.float64],
    Xy: NDArray[np.float64],
    n_nonzero_coefs: int,
    norms_squared: NDArray[np.float64] | None,
    tol: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    xy_2d = Xy.reshape(-1, 1) if Xy.ndim == 1 else Xy
    coefficients = np.zeros((Gram.shape[0], xy_2d.shape[1]), dtype=np.float64)
    n_iters = np.zeros(xy_2d.shape[1], dtype=np.int64)
    for target_index in range(xy_2d.shape[1]):
        tol_0 = None if norms_squared is None else float(norms_squared[target_index])
        coef, n_iter = _gram_omp_solve_single(Gram, xy_2d[:, target_index], n_nonzero_coefs, tol_0, tol)
        coefficients[:, target_index] = coef
        n_iters[target_index] = n_iter
    if Xy.ndim == 1:
        return coefficients[:, 0], n_iters
    return coefficients, n_iters


def _resolve_omp_cv_max_iter(max_iter: int | None, n_features: int) -> int:
    if max_iter is not None:
        return int(max_iter)
    return min(max(int(0.1 * n_features), 5), n_features)


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


@register_atom(witness_orthogonal_mp)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda n_nonzero_coefs, X: _omp_n_nonzero_valid(n_nonzero_coefs, np.asarray(X).shape[1]), "n_nonzero_coefs must be positive and no larger than feature count")
@icontract.require(lambda tol: _omp_tol_valid(tol), "tol must be None or a non-negative finite value")
@icontract.require(lambda precompute: _omp_precompute_valid(precompute), "precompute must be boolean or auto")
@icontract.require(lambda copy_X: _bool_value(copy_X), "copy_X must be boolean")
@icontract.require(lambda return_path: return_path is False, "return_path is outside this atom scope")
@icontract.require(lambda return_n_iter: return_n_iter is False, "return_n_iter is outside this atom scope")
@icontract.ensure(lambda result, X, y: _omp_coefficients_valid(result, X, y), "OMP coefficients must match feature and target dimensions")
def orthogonal_mp(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    n_nonzero_coefs: int | None = None,
    tol: float | None = None,
    precompute: bool | str = False,
    copy_X: bool = True,
    return_path: bool = False,
    return_n_iter: bool = False,
) -> NDArray[np.float64]:
    """Solve dense orthogonal matching pursuit coefficients from samples."""
    del copy_X, return_path, return_n_iter
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    if precompute == "auto":
        precompute = checked_x.shape[0] > checked_x.shape[1]
    resolved_n_nonzero = _resolve_omp_n_nonzero(n_nonzero_coefs, tol, checked_x.shape[1])
    if precompute:
        gram = np.asarray(checked_x.T @ checked_x, dtype=np.float64)
        xy = np.asarray(checked_x.T @ (checked_y.reshape(-1, 1) if checked_y.ndim == 1 else checked_y), dtype=np.float64)
        norms_squared = None if tol is None else np.sum((checked_y.reshape(-1, 1) if checked_y.ndim == 1 else checked_y) ** 2, axis=0)
        result = orthogonal_mp_gram(
            gram,
            xy[:, 0] if checked_y.ndim == 1 else xy,
            n_nonzero_coefs=resolved_n_nonzero,
            tol=tol,
            norms_squared=None if norms_squared is None else np.asarray(norms_squared, dtype=np.float64),
        )
        return np.asarray(result, dtype=np.float64)
    coefficients, _ = _omp_solve(checked_x, checked_y, resolved_n_nonzero, tol)
    return np.asarray(coefficients, dtype=np.float64)


@register_atom(witness_orthogonal_mp_gram)
@icontract.require(lambda Gram: _square_matrix(Gram), "Gram must be square")
@icontract.require(lambda Gram, Xy: _gram_and_xy_match(Gram, Xy), "Xy must match Gram feature count")
@icontract.require(lambda Gram, Xy: _finite_gram_inputs(Gram, Xy), "Gram and Xy must contain finite numeric values")
@icontract.require(lambda n_nonzero_coefs, Gram: _omp_n_nonzero_valid(n_nonzero_coefs, np.asarray(Gram).shape[0]), "n_nonzero_coefs must be positive and no larger than feature count")
@icontract.require(lambda tol: _omp_tol_valid(tol), "tol must be None or a non-negative finite value")
@icontract.require(lambda norms_squared, Xy, tol: _omp_norms_valid(norms_squared, Xy, tol), "norms_squared must be supplied for tol mode and match target count")
@icontract.require(lambda copy_Gram: _bool_value(copy_Gram), "copy_Gram must be boolean")
@icontract.require(lambda copy_Xy: _bool_value(copy_Xy), "copy_Xy must be boolean")
@icontract.require(lambda return_path: return_path is False, "return_path is outside this atom scope")
@icontract.require(lambda return_n_iter: return_n_iter is False, "return_n_iter is outside this atom scope")
@icontract.ensure(lambda result, Gram, Xy: _omp_gram_coefficients_valid(result, Gram, Xy), "OMP Gram coefficients must match feature and target dimensions")
def orthogonal_mp_gram(
    Gram: NDArray[np.float64],
    Xy: NDArray[np.float64],
    *,
    n_nonzero_coefs: int | None = None,
    tol: float | None = None,
    norms_squared: tuple[float, ...] | NDArray[np.float64] | None = None,
    copy_Gram: bool = True,
    copy_Xy: bool = True,
    return_path: bool = False,
    return_n_iter: bool = False,
) -> NDArray[np.float64]:
    """Solve dense orthogonal matching pursuit coefficients from Gram inputs."""
    del copy_Gram, copy_Xy, return_path, return_n_iter
    checked_gram = check_array(Gram, dtype=np.float64, ensure_2d=True)
    checked_xy = np.asarray(Xy, dtype=np.float64)
    resolved_n_nonzero = _resolve_omp_n_nonzero(n_nonzero_coefs, tol, checked_gram.shape[0], gram_default=True)
    if resolved_n_nonzero <= 0:
        raise ValueError("n_nonzero_coefs must be positive")
    norms = None
    if norms_squared is not None:
        norms = np.atleast_1d(np.asarray(norms_squared, dtype=np.float64))
        xy_targets = 1 if checked_xy.ndim == 1 else checked_xy.shape[1]
        if norms.shape[0] == 1 and xy_targets > 1:
            norms = np.full(xy_targets, norms[0], dtype=np.float64)
    coefficients, _ = _gram_omp_solve(checked_gram, checked_xy, resolved_n_nonzero, norms, tol)
    return np.asarray(coefficients, dtype=np.float64)


@register_atom(witness_orthogonal_matching_pursuit_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda n_nonzero_coefs, X: _omp_n_nonzero_valid(n_nonzero_coefs, np.asarray(X).shape[1]), "n_nonzero_coefs must be positive and no larger than feature count")
@icontract.require(lambda tol: _omp_tol_valid(tol), "tol must be None or a non-negative finite value")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda precompute: _omp_precompute_valid(precompute), "precompute must be boolean or auto")
@icontract.ensure(lambda result: _omp_state_valid(result), "OMP state must contain finite fitted coefficients")
def orthogonal_matching_pursuit_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    n_nonzero_coefs: int | None = None,
    tol: float | None = None,
    fit_intercept: bool = True,
    precompute: bool | str = "auto",
) -> OrthogonalMatchingPursuitState:
    """Fit dense orthogonal matching pursuit coefficients and intercept."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    y_was_1d = checked_y.ndim == 1
    checked_y_2d = checked_y.reshape(-1, 1) if y_was_1d else checked_y
    centered_x, centered_y, x_offset, y_offset = _center_without_rescale(checked_x, checked_y_2d, fit_intercept, None)
    resolved_n_nonzero = _resolve_omp_n_nonzero(n_nonzero_coefs, tol, checked_x.shape[1])
    use_precompute = centered_x.shape[0] > centered_x.shape[1] if precompute == "auto" else bool(precompute)
    if use_precompute:
        gram = np.asarray(centered_x.T @ centered_x, dtype=np.float64)
        xy = np.asarray(centered_x.T @ centered_y, dtype=np.float64)
        norms = None if tol is None else np.sum(centered_y**2, axis=0)
        coefficients_by_feature, n_iters = _gram_omp_solve(gram, xy, resolved_n_nonzero, norms, tol)
    else:
        coefficients_by_feature, n_iters = _omp_solve(centered_x, centered_y, resolved_n_nonzero, tol)
    coef_2d = coefficients_by_feature.reshape(-1, 1) if coefficients_by_feature.ndim == 1 else coefficients_by_feature
    intercept = y_offset - x_offset @ coef_2d if fit_intercept else np.zeros(checked_y_2d.shape[1], dtype=np.float64)
    coef_state = coef_2d[:, 0] if y_was_1d else coef_2d.T
    return OrthogonalMatchingPursuitState(
        coef=np.asarray(coef_state, dtype=np.float64),
        intercept=np.atleast_1d(np.asarray(intercept, dtype=np.float64)),
        n_iter=np.asarray(n_iters, dtype=np.int64),
        n_nonzero_coefs=None if tol is not None else resolved_n_nonzero,
        tol=None if tol is None else float(tol),
        fit_intercept=fit_intercept,
        precompute=precompute,
        n_features_in=int(checked_x.shape[1]),
        n_outputs=int(checked_y_2d.shape[1]),
    )


@register_atom(witness_orthogonal_matching_pursuit_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _omp_feature_count_matches(X, state), "X feature count must match fitted OMP state")
@icontract.require(lambda state: _omp_state_valid(state), "state must be a fitted OMP state")
@icontract.ensure(lambda result, X, state: _omp_prediction_valid(result, X, state), "predictions must match fitted output width")
def orthogonal_matching_pursuit_predict(X: NDArray[np.float64], state: OrthogonalMatchingPursuitState) -> NDArray[np.float64]:
    """Predict dense outputs from fitted orthogonal matching pursuit coefficients."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    if state.n_outputs == 1:
        return np.asarray(checked_x @ state.coef + state.intercept[0], dtype=np.float64)
    return np.asarray(checked_x @ state.coef.T + state.intercept, dtype=np.float64)


@register_atom(witness_omp_path_residues)
@icontract.require(lambda X_train: _matrix_2d(X_train), "X_train must be 2D")
@icontract.require(lambda X_test: _matrix_2d(X_test), "X_test must be 2D")
@icontract.require(lambda y_train: _target_1d(y_train), "y_train must be 1D")
@icontract.require(lambda y_test: _target_1d(y_test), "y_test must be 1D")
@icontract.require(lambda X_train, y_train: _same_sample_count(X_train, y_train), "training X and y must have matching sample counts")
@icontract.require(lambda X_test, y_test: _same_sample_count(X_test, y_test), "test X and y must have matching sample counts")
@icontract.require(lambda X_train, X_test: np.asarray(X_train).shape[1] == np.asarray(X_test).shape[1], "train and test feature counts must match")
@icontract.require(lambda X_train, y_train: _finite_inputs(X_train, y_train), "training inputs must be finite")
@icontract.require(lambda X_test, y_test: _finite_inputs(X_test, y_test), "test inputs must be finite")
@icontract.require(lambda copy: _bool_value(copy), "copy must be boolean")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter, X_train: _omp_cv_max_iter_valid(max_iter, np.asarray(X_train).shape[1]), "max_iter must be positive and no larger than feature count")
@icontract.ensure(lambda result, X_test, max_iter: _omp_cv_residues_valid(result, X_test, max_iter), "residual path must match max_iter and held-out sample count")
def omp_path_residues(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    *,
    copy: bool = True,
    fit_intercept: bool = True,
    max_iter: int = 100,
) -> NDArray[np.float64]:
    """Compute held-out residuals along a dense OMP coefficient path."""
    del copy
    train_x = check_array(X_train, dtype=np.float64, ensure_2d=True)
    test_x = check_array(X_test, dtype=np.float64, ensure_2d=True)
    train_y = check_array(y_train, dtype=np.float64, ensure_2d=False, input_name="y_train")
    test_y = check_array(y_test, dtype=np.float64, ensure_2d=False, input_name="y_test")
    if fit_intercept:
        x_mean = np.mean(train_x, axis=0)
        y_mean = float(np.mean(train_y))
        train_x = train_x - x_mean
        test_x = test_x - x_mean
        train_y = train_y - y_mean
        test_y = test_y - y_mean
    path, n_steps = _omp_path_coefficients_single(train_x, train_y, max_iter)
    path_length = max(n_steps, 1)
    residues = np.zeros((path_length, test_x.shape[0]), dtype=np.float64)
    if n_steps > 0:
        residues[:n_steps] = path[:, :n_steps].T @ test_x.T - test_y
    else:
        residues[0] = -test_y
    return residues


@register_atom(witness_orthogonal_matching_pursuit_cv_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda copy: _bool_value(copy), "copy must be boolean")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter, X: _omp_cv_max_iter_valid(max_iter, np.asarray(X).shape[1]), "max_iter must be positive and no larger than feature count")
@icontract.require(lambda cv, X: _omp_cv_valid(cv, np.asarray(X).shape[0]), "cv must be None or an integer between 2 and sample count")
@icontract.require(lambda n_jobs: n_jobs is None, "parallel n_jobs is outside this atom scope")
@icontract.require(lambda verbose: verbose is False, "verbose output is outside this atom scope")
@icontract.ensure(lambda result: _omp_cv_state_valid(result), "OMP CV state must contain finite fitted coefficients")
def orthogonal_matching_pursuit_cv_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    copy: bool = True,
    fit_intercept: bool = True,
    max_iter: int | None = None,
    cv: int | None = None,
    n_jobs: None = None,
    verbose: bool = False,
) -> OrthogonalMatchingPursuitCVState:
    """Fit dense OMP after KFold selection of the active feature count."""
    del n_jobs, verbose
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    resolved_max_iter = _resolve_omp_cv_max_iter(max_iter, checked_x.shape[1])
    resolved_cv = 5 if cv is None else int(cv)
    splitter = KFold(n_splits=resolved_cv, shuffle=False)
    cv_paths = [
        omp_path_residues(
            checked_x[train],
            checked_y[train],
            checked_x[test],
            checked_y[test],
            copy=copy,
            fit_intercept=fit_intercept,
            max_iter=resolved_max_iter,
        )
        for train, test in splitter.split(checked_x)
    ]
    min_early_stop = min(path.shape[0] for path in cv_paths)
    mse_folds = np.asarray([(path[:min_early_stop] ** 2).mean(axis=1) for path in cv_paths], dtype=np.float64)
    mean_mse = mse_folds.mean(axis=0)
    best_candidates = np.flatnonzero(np.isclose(mean_mse, np.min(mean_mse), rtol=1e-12, atol=1e-12))
    best_n_nonzero = int(best_candidates[0] + 1)
    state = orthogonal_matching_pursuit_fit(
        checked_x,
        checked_y,
        n_nonzero_coefs=best_n_nonzero,
        fit_intercept=fit_intercept,
        precompute="auto",
    )
    return OrthogonalMatchingPursuitCVState(
        coef=np.asarray(state.coef, dtype=np.float64),
        intercept=state.intercept,
        n_iter=state.n_iter,
        n_nonzero_coefs=best_n_nonzero,
        max_iter=resolved_max_iter,
        cv=resolved_cv,
        fit_intercept=fit_intercept,
        n_features_in=state.n_features_in,
    )


@register_atom(witness_orthogonal_matching_pursuit_cv_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _omp_cv_feature_count_matches(X, state), "X feature count must match fitted OMP CV state")
@icontract.require(lambda state: _omp_cv_state_valid(state), "state must be a fitted OMP CV state")
@icontract.ensure(lambda result, X: _omp_cv_prediction_valid(result, X), "predictions must match sample count")
def orthogonal_matching_pursuit_cv_predict(X: NDArray[np.float64], state: OrthogonalMatchingPursuitCVState) -> NDArray[np.float64]:
    """Predict dense outputs from fitted cross-validated OMP coefficients."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    return np.asarray(checked_x @ state.coef + state.intercept[0], dtype=np.float64)


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


@register_atom(witness_ridge_classifier_cv_scores)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X: _sample_count_at_least_two(X), "leave-one-out CV requires at least two samples")
@icontract.require(lambda X, y: _finite_classifier_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda y: _class_count_at_least_two(y), "ridge classifier CV requires at least two classes")
@icontract.require(lambda alphas: _alphas_strictly_positive(alphas), "alphas must be finite and strictly positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda scoring: scoring is None, "only default negative MSE scoring is covered")
@icontract.require(lambda cv: cv is None, "only leave-one-out CV is covered")
@icontract.require(lambda class_weight: class_weight is None, "class_weight is outside this RidgeClassifierCV atom scope")
@icontract.require(lambda sample_weight: sample_weight is None, "sample_weight is outside this RidgeClassifierCV atom scope")
@icontract.ensure(lambda result, alphas: _ridge_cv_scores_valid(result, alphas), "CV scores must match the alpha grid")
def ridge_classifier_cv_scores(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    alphas: float | tuple[float, ...] | NDArray[np.float64] = (0.1, 1.0, 10.0),
    *,
    fit_intercept: bool = True,
    scoring: None = None,
    cv: None = None,
    class_weight: None = None,
    sample_weight: None = None,
) -> NDArray[np.float64]:
    """Compute dense leave-one-out RidgeClassifierCV scores for each alpha."""
    del scoring, cv, class_weight, sample_weight
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    classes = np.unique(checked_y)
    encoded_y = _binarize_ridge_classes(checked_y, classes)
    alpha_values = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    scores = np.empty(alpha_values.shape[0], dtype=np.float64)
    sample_indices = np.arange(checked_x.shape[0])
    for alpha_index, current_alpha in enumerate(alpha_values):
        predictions = np.empty_like(encoded_y, dtype=np.float64)
        for held_out in range(checked_x.shape[0]):
            train_mask = sample_indices != held_out
            train_y = encoded_y[train_mask, 0] if encoded_y.shape[1] == 1 else encoded_y[train_mask]
            state = ridge_fit(
                checked_x[train_mask],
                train_y,
                alpha=float(current_alpha),
                fit_intercept=fit_intercept,
                solver="cholesky",
            )
            prediction = ridge_predict(checked_x[held_out : held_out + 1], state)
            predictions[held_out] = np.asarray(prediction, dtype=np.float64).reshape(1, -1)[0]
        scores[alpha_index] = -float(np.mean((encoded_y - predictions) ** 2))
    return scores


@register_atom(witness_ridge_classifier_cv_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X: _sample_count_at_least_two(X), "leave-one-out CV requires at least two samples")
@icontract.require(lambda X, y: _finite_classifier_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda y: _class_count_at_least_two(y), "ridge classifier CV requires at least two classes")
@icontract.require(lambda alphas: _alphas_strictly_positive(alphas), "alphas must be finite and strictly positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda scoring: scoring is None, "only default negative MSE scoring is covered")
@icontract.require(lambda cv: cv is None, "only leave-one-out CV is covered")
@icontract.require(lambda class_weight: class_weight is None, "class_weight is outside this RidgeClassifierCV atom scope")
@icontract.require(lambda store_cv_results: store_cv_results is False, "store_cv_results is outside this atom scope")
@icontract.require(lambda sample_weight: sample_weight is None, "sample_weight is outside this RidgeClassifierCV atom scope")
@icontract.ensure(lambda result: _ridge_classifier_cv_state_valid(result), "RidgeClassifierCV state must contain fitted coefficients and selected alpha")
def ridge_classifier_cv_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    alphas: float | tuple[float, ...] | NDArray[np.float64] = (0.1, 1.0, 10.0),
    fit_intercept: bool = True,
    scoring: None = None,
    cv: None = None,
    class_weight: None = None,
    store_cv_results: bool = False,
    sample_weight: None = None,
) -> RidgeClassifierCVState:
    """Fit dense ridge classification after leave-one-out alpha selection."""
    del scoring, cv, class_weight, store_cv_results, sample_weight
    alpha_values = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    scores = ridge_classifier_cv_scores(X, y, alpha_values, fit_intercept=fit_intercept)
    best_index = int(np.argmax(scores))
    selected_alpha = float(alpha_values[best_index])
    classifier_state = ridge_classifier_fit(X, y, alpha=selected_alpha, fit_intercept=fit_intercept, solver="cholesky")
    return RidgeClassifierCVState(
        coef=classifier_state.coef,
        intercept=classifier_state.intercept,
        classes=classifier_state.classes,
        alpha=np.asarray([selected_alpha], dtype=np.float64),
        best_score=np.asarray([scores[best_index]], dtype=np.float64),
        fit_intercept=fit_intercept,
        n_features_in=classifier_state.n_features_in,
    )


@register_atom(witness_ridge_classifier_cv_decision_function)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _ridge_classifier_cv_feature_count_matches(X, state), "X feature count must match fitted RidgeClassifierCV state")
@icontract.require(lambda state: _ridge_classifier_cv_state_valid(state), "state must be a fitted RidgeClassifierCV state")
@icontract.ensure(lambda result, X, state: _ridge_classifier_cv_scores_output_valid(result, X, state), "decision scores must match fitted class width")
def ridge_classifier_cv_decision_function(X: NDArray[np.float64], state: RidgeClassifierCVState) -> NDArray[np.float64]:
    """Compute dense RidgeClassifierCV confidence scores."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    scores = checked_x @ state.coef.T + state.intercept
    if state.classes.shape[0] == 2:
        return np.asarray(scores.reshape(-1), dtype=np.float64)
    return np.asarray(scores, dtype=np.float64)


@register_atom(witness_ridge_classifier_cv_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _ridge_classifier_cv_feature_count_matches(X, state), "X feature count must match fitted RidgeClassifierCV state")
@icontract.require(lambda state: _ridge_classifier_cv_state_valid(state), "state must be a fitted RidgeClassifierCV state")
@icontract.ensure(lambda result, X, state: _ridge_classifier_cv_prediction_valid(result, X, state), "predictions must be fitted class labels")
def ridge_classifier_cv_predict(X: NDArray[np.float64], state: RidgeClassifierCVState) -> NDArray[np.float64]:
    """Predict dense RidgeClassifierCV labels."""
    scores = ridge_classifier_cv_decision_function(X, state)
    if scores.ndim == 1:
        indices = (scores > 0.0).astype(np.int64)
    else:
        indices = np.argmax(scores, axis=1)
    return np.asarray(state.classes[indices], dtype=np.float64)


def _ard_keep_mask(state: ARDRegressionState) -> NDArray[np.bool_]:
    return np.asarray(state.lambda_ < state.threshold_lambda, dtype=np.bool_)


def _ard_regression_state_valid(state: ARDRegressionState) -> bool:
    keep = _ard_keep_mask(state)
    return bool(
        state.coef.shape == (state.n_features_in,)
        and state.lambda_.shape == (state.n_features_in,)
        and state.sigma.shape == (int(np.sum(keep)), int(np.sum(keep)))
        and state.scores.ndim == 1
        and state.x_offset.shape == (state.n_features_in,)
        and state.x_scale.shape == (state.n_features_in,)
        and state.n_iter >= 1
        and state.alpha > 0.0
        and state.threshold_lambda > 0.0
        and isinstance(state.fit_intercept, bool)
        and isinstance(state.compute_score, bool)
        and np.isfinite(state.intercept)
        and np.isfinite(state.alpha)
        and np.isfinite(state.threshold_lambda)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.lambda_))
        and np.all(state.lambda_ > 0.0)
        and np.all(np.isfinite(state.sigma))
        and np.allclose(state.sigma, state.sigma.T)
        and np.all(np.isfinite(state.scores))
        and np.all(np.isfinite(state.x_offset))
        and np.all(np.isfinite(state.x_scale))
    )


def _ard_regression_feature_count_matches(X: NDArray[np.float64], state: ARDRegressionState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _ard_regression_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isfinite(values)))


def _ard_update_sigma(
    X: NDArray[np.float64],
    alpha: float,
    lambda_values: NDArray[np.float64],
    keep_lambda: NDArray[np.bool_],
) -> NDArray[np.float64]:
    X_keep = X[:, keep_lambda]
    if X.shape[0] >= X.shape[1]:
        gram = np.dot(X_keep.T, X_keep)
        eye = np.eye(gram.shape[0], dtype=np.float64)
        sigma_inv = lambda_values[keep_lambda] * eye + alpha * gram
        return np.asarray(linalg.pinvh(sigma_inv), dtype=np.float64)
    inv_lambda = 1.0 / lambda_values[keep_lambda].reshape(1, -1)
    sigma = linalg.pinvh(np.eye(X.shape[0], dtype=np.float64) / alpha + np.dot(X_keep * inv_lambda, X_keep.T))
    sigma = np.dot(sigma, X_keep * inv_lambda)
    sigma = -np.dot(inv_lambda.reshape(-1, 1) * X_keep.T, sigma)
    sigma[np.diag_indices(sigma.shape[1])] += 1.0 / lambda_values[keep_lambda]
    return np.asarray(sigma, dtype=np.float64)


def _ard_update_coefficients(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    coef: NDArray[np.float64],
    alpha: float,
    keep_lambda: NDArray[np.bool_],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    updated = coef.copy()
    updated[keep_lambda] = alpha * np.linalg.multi_dot([sigma, X[:, keep_lambda].T, y])
    return np.asarray(updated, dtype=np.float64)


def _ard_logdet(matrix: NDArray[np.float64]) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:
        return float("-inf")
    return float(value)


@register_atom(witness_ard_regression_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X: _sample_count_at_least_two(X), "X must contain at least two samples")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda max_iter: isinstance(max_iter, int) and not isinstance(max_iter, bool) and max_iter >= 1, "max_iter must be positive")
@icontract.require(lambda tol: _positive_finite(tol), "tol must be positive")
@icontract.require(lambda alpha_1: _nonnegative_finite(alpha_1), "alpha_1 must be non-negative")
@icontract.require(lambda alpha_2: _nonnegative_finite(alpha_2), "alpha_2 must be non-negative")
@icontract.require(lambda lambda_1: _nonnegative_finite(lambda_1), "lambda_1 must be non-negative")
@icontract.require(lambda lambda_2: _nonnegative_finite(lambda_2), "lambda_2 must be non-negative")
@icontract.require(lambda compute_score: _bool_value(compute_score), "compute_score must be boolean")
@icontract.require(lambda threshold_lambda: _positive_finite(threshold_lambda), "threshold_lambda must be positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda copy_X: _bool_value(copy_X), "copy_X must be boolean")
@icontract.require(lambda verbose: verbose is False, "verbose output is outside this atom scope")
@icontract.ensure(lambda result: _ard_regression_state_valid(result), "ARD state must contain finite posterior parameters")
def ard_regression_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    compute_score: bool = False,
    threshold_lambda: float = 1e4,
    fit_intercept: bool = True,
    copy_X: bool = True,
    verbose: bool = False,
) -> ARDRegressionState:
    """Fit dense automatic relevance determination posterior parameters."""
    del copy_X, verbose
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    n_samples, n_features = checked_x.shape
    coef = np.zeros(n_features, dtype=np.float64)
    centered_x_2d, centered_y_2d, x_offset, y_offset = _center_and_rescale(
        checked_x,
        checked_y.reshape(-1, 1),
        fit_intercept,
        None,
    )
    centered_y = np.ravel(centered_y_2d)
    keep_lambda = np.ones(n_features, dtype=bool)
    eps = np.finfo(np.float64).eps
    alpha = float(1.0 / (np.var(centered_y) + eps))
    lambda_values = np.ones(n_features, dtype=np.float64)
    scores: list[float] = []
    coef_old: NDArray[np.float64] | None = None
    sigma = np.empty((n_features, n_features), dtype=np.float64)

    iteration = 0
    for iteration in range(max_iter):
        sigma = _ard_update_sigma(centered_x_2d, alpha, lambda_values, keep_lambda)
        coef = _ard_update_coefficients(centered_x_2d, centered_y, coef, alpha, keep_lambda, sigma)
        sse = float(np.sum((centered_y - np.dot(centered_x_2d, coef)) ** 2))
        gamma = 1.0 - lambda_values[keep_lambda] * np.diag(sigma)
        lambda_values[keep_lambda] = (gamma + 2.0 * lambda_1) / (coef[keep_lambda] ** 2 + 2.0 * lambda_2)
        alpha = float((n_samples - float(np.sum(gamma)) + 2.0 * alpha_1) / (sse + 2.0 * alpha_2))
        keep_lambda = lambda_values < threshold_lambda
        coef[~keep_lambda] = 0.0

        if compute_score:
            score = float(np.sum(lambda_1 * np.log(lambda_values) - lambda_2 * lambda_values))
            score += alpha_1 * log(alpha) - alpha_2 * alpha
            score += 0.5 * (
                _ard_logdet(sigma)
                + n_samples * log(alpha)
                + float(np.sum(np.log(lambda_values)))
                - alpha * sse
                - float(np.sum(lambda_values * coef**2))
            )
            scores.append(float(score))

        if iteration > 0 and coef_old is not None and np.sum(np.abs(coef_old - coef)) < tol:
            break
        coef_old = np.copy(coef)
        if not bool(np.any(keep_lambda)):
            break

    n_iter = int(iteration + 1)
    if bool(np.any(keep_lambda)):
        sigma = _ard_update_sigma(centered_x_2d, alpha, lambda_values, keep_lambda)
        coef = _ard_update_coefficients(centered_x_2d, centered_y, coef, alpha, keep_lambda, sigma)
    else:
        sigma = np.empty((0, 0), dtype=np.float64)
    intercept = float(y_offset[0] - np.dot(x_offset, coef)) if fit_intercept else 0.0
    return ARDRegressionState(
        coef=np.asarray(coef, dtype=np.float64),
        intercept=intercept,
        alpha=float(alpha),
        lambda_=np.asarray(lambda_values, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        n_iter=n_iter,
        threshold_lambda=float(threshold_lambda),
        x_offset=np.asarray(x_offset, dtype=np.float64),
        x_scale=np.ones(n_features, dtype=np.float64),
        fit_intercept=fit_intercept,
        compute_score=compute_score,
        n_features_in=int(n_features),
    )


@register_atom(witness_ard_regression_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _ard_regression_feature_count_matches(X, state), "X feature count must match fitted ARD state")
@icontract.require(lambda state: _ard_regression_state_valid(state), "state must be a fitted ARD state")
@icontract.require(lambda return_std: return_std is False, "use ard_regression_predict_std for posterior standard deviations")
@icontract.ensure(lambda result, X: _ard_regression_prediction_valid(result, X), "predictions must be finite per-row values")
def ard_regression_predict(
    X: NDArray[np.float64],
    state: ARDRegressionState,
    *,
    return_std: bool = False,
) -> NDArray[np.float64]:
    """Predict posterior mean values from fitted ARD state."""
    del return_std
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    return np.asarray(np.dot(checked_x, state.coef) + state.intercept, dtype=np.float64)


@register_atom(witness_ard_regression_predict_std)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _ard_regression_feature_count_matches(X, state), "X feature count must match fitted ARD state")
@icontract.require(lambda state: _ard_regression_state_valid(state), "state must be a fitted ARD state")
@icontract.ensure(lambda result, X: _ard_regression_prediction_valid(result, X), "standard deviations must be finite per-row values")
def ard_regression_predict_std(X: NDArray[np.float64], state: ARDRegressionState) -> NDArray[np.float64]:
    """Predict posterior standard deviations from fitted ARD state."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    keep = _ard_keep_mask(state)
    checked_x = checked_x[:, keep]
    sigmas_squared = (np.dot(checked_x, state.sigma) * checked_x).sum(axis=1)
    return np.asarray(np.sqrt(sigmas_squared + (1.0 / state.alpha)), dtype=np.float64)


def _theil_sen_state_valid(state: TheilSenRegressorState) -> bool:
    return bool(
        state.coef.shape == (state.n_features_in,)
        and np.all(np.isfinite(state.coef))
        and np.isfinite(state.intercept)
        and np.isfinite(state.breakdown)
        and 0.0 <= state.breakdown <= 0.5
        and state.n_iter >= 0
        and state.n_subpopulation >= 1
        and state.n_subsamples >= 1
        and isinstance(state.fit_intercept, bool)
        and state.max_subpopulation >= 1
        and state.n_features_in >= 1
    )


def _theil_sen_feature_count_matches(X: NDArray[np.float64], state: TheilSenRegressorState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _theil_sen_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isfinite(values)))


def _random_state_valid(random_state: int | None) -> bool:
    return random_state is None or (isinstance(random_state, int) and not isinstance(random_state, bool) and random_state >= 0)


def _theil_sen_n_jobs_valid(n_jobs: int | None) -> bool:
    return n_jobs is None or n_jobs == 1


def _theil_sen_subsample_params(
    n_samples: int,
    n_features: int,
    *,
    fit_intercept: bool,
    n_subsamples: int | None,
    max_subpopulation: int,
) -> tuple[int, int]:
    n_dim = n_features + int(fit_intercept)
    if n_subsamples is not None:
        if n_subsamples > n_samples:
            raise ValueError("n_subsamples must not exceed n_samples")
        if n_samples >= n_features:
            if n_dim > n_subsamples:
                raise ValueError("n_subsamples must cover the feature dimension")
        elif n_subsamples != n_samples:
            raise ValueError("n_subsamples must equal n_samples when n_samples < n_features")
    else:
        n_subsamples = min(n_dim, n_samples)
    all_combinations = max(1, int(np.rint(binom(n_samples, n_subsamples))))
    return int(n_subsamples), int(min(max_subpopulation, all_combinations))


def _theil_sen_breakdown_point(n_samples: int, n_subsamples: int) -> float:
    return float(1.0 - (0.5 ** (1.0 / n_subsamples) * (n_samples - n_subsamples + 1) + n_subsamples - 1) / n_samples)


def _theil_sen_lstsq(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    indices: NDArray[np.int64],
    fit_intercept: bool,
) -> NDArray[np.float64]:
    intercept_width = int(fit_intercept)
    n_features = X.shape[1] + intercept_width
    n_subsamples = indices.shape[1]
    weights = np.empty((indices.shape[0], n_features), dtype=np.float64)
    X_subpopulation = np.ones((n_subsamples, n_features), dtype=np.float64)
    y_subpopulation = np.zeros(max(n_subsamples, n_features), dtype=np.float64)
    (lstsq,) = get_lapack_funcs(("gelss",), (X_subpopulation, y_subpopulation))
    for index, subset in enumerate(indices):
        X_subpopulation[:, intercept_width:] = X[subset, :]
        y_subpopulation[:n_subsamples] = y[subset]
        weights[index] = lstsq(X_subpopulation, y_subpopulation)[1][:n_features]
    return weights


def _theil_sen_modified_weiszfeld_step(X: NDArray[np.float64], x_old: NDArray[np.float64]) -> NDArray[np.float64]:
    epsilon = np.finfo(np.float64).eps
    diff = X - x_old
    diff_norm = np.sqrt(np.sum(diff**2, axis=1))
    mask = diff_norm >= epsilon
    is_x_old_in_x = int(np.sum(mask) < X.shape[0])
    diff_norm = diff_norm[mask][:, np.newaxis]
    quotient_norm = linalg.norm(np.sum(diff[mask] / diff_norm, axis=0))
    if quotient_norm > epsilon:
        new_direction = np.sum(X[mask, :] / diff_norm, axis=0) / np.sum(1.0 / diff_norm, axis=0)
    else:
        new_direction = 1.0
        quotient_norm = 1.0
    return np.asarray(
        max(0.0, 1.0 - is_x_old_in_x / quotient_norm) * new_direction
        + min(1.0, is_x_old_in_x / quotient_norm) * x_old,
        dtype=np.float64,
    )


def _theil_sen_spatial_median(
    X: NDArray[np.float64],
    *,
    max_iter: int = 300,
    tol: float = 1.0e-3,
) -> tuple[int, NDArray[np.float64]]:
    if X.shape[1] == 1:
        return 1, np.asarray(np.median(X.ravel(), keepdims=True), dtype=np.float64)
    squared_tol = tol**2
    spatial_median_old = np.mean(X, axis=0)
    n_iter = 0
    for n_iter in range(max_iter):
        spatial_median = _theil_sen_modified_weiszfeld_step(X, spatial_median_old)
        if np.sum((spatial_median_old - spatial_median) ** 2) < squared_tol:
            break
        spatial_median_old = spatial_median
    return int(n_iter), np.asarray(spatial_median, dtype=np.float64)


@register_atom(witness_theil_sen_regressor_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X: _sample_count_at_least_two(X), "X must contain at least two samples")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_subpopulation: isinstance(max_subpopulation, (int, float)) and not isinstance(max_subpopulation, bool) and np.isfinite(float(max_subpopulation)) and int(max_subpopulation) >= 1, "max_subpopulation must be positive")
@icontract.require(lambda n_subsamples: n_subsamples is None or (isinstance(n_subsamples, int) and not isinstance(n_subsamples, bool) and n_subsamples >= 1), "n_subsamples must be positive when provided")
@icontract.require(lambda max_iter: isinstance(max_iter, int) and not isinstance(max_iter, bool) and max_iter >= 1, "max_iter must be positive")
@icontract.require(lambda tol: _positive_finite(tol), "tol must be positive")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be a non-negative integer or None")
@icontract.require(lambda n_jobs: _theil_sen_n_jobs_valid(n_jobs), "only single-process n_jobs is covered")
@icontract.require(lambda verbose: verbose is False, "verbose output is outside this atom scope")
@icontract.ensure(lambda result: _theil_sen_state_valid(result), "Theil-Sen state must contain finite fitted coefficients")
def theil_sen_regressor_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    fit_intercept: bool = True,
    max_subpopulation: int | float = 10000,
    n_subsamples: int | None = None,
    max_iter: int = 300,
    tol: float = 1.0e-3,
    random_state: int | None = None,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> TheilSenRegressorState:
    """Fit dense single-output Theil-Sen regression coefficients."""
    del n_jobs, verbose
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    n_samples, n_features = checked_x.shape
    max_subpopulation_int = int(max_subpopulation)
    resolved_n_subsamples, n_subpopulation = _theil_sen_subsample_params(
        n_samples,
        n_features,
        fit_intercept=fit_intercept,
        n_subsamples=n_subsamples,
        max_subpopulation=max_subpopulation_int,
    )
    all_combinations = int(np.rint(binom(n_samples, resolved_n_subsamples)))
    if all_combinations <= max_subpopulation_int:
        indices = np.asarray(list(combinations(range(n_samples), resolved_n_subsamples)), dtype=np.int64)
    else:
        rng = np.random.RandomState(random_state)
        indices = np.asarray(
            [rng.choice(n_samples, size=resolved_n_subsamples, replace=False) for _ in range(n_subpopulation)],
            dtype=np.int64,
        )
    weights = _theil_sen_lstsq(checked_x, checked_y, indices, fit_intercept)
    n_iter, coefs = _theil_sen_spatial_median(weights, max_iter=max_iter, tol=tol)
    if fit_intercept:
        intercept = float(coefs[0])
        coef = np.asarray(coefs[1:], dtype=np.float64)
    else:
        intercept = 0.0
        coef = np.asarray(coefs, dtype=np.float64)
    return TheilSenRegressorState(
        coef=coef,
        intercept=intercept,
        breakdown=_theil_sen_breakdown_point(n_samples, resolved_n_subsamples),
        n_iter=int(n_iter),
        n_subpopulation=int(n_subpopulation),
        n_subsamples=int(resolved_n_subsamples),
        fit_intercept=fit_intercept,
        max_subpopulation=max_subpopulation_int,
        n_features_in=int(n_features),
    )


@register_atom(witness_theil_sen_regressor_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _theil_sen_feature_count_matches(X, state), "X feature count must match fitted Theil-Sen state")
@icontract.require(lambda state: _theil_sen_state_valid(state), "state must be a fitted Theil-Sen state")
@icontract.ensure(lambda result, X: _theil_sen_prediction_valid(result, X), "predictions must be finite per-row values")
def theil_sen_regressor_predict(X: NDArray[np.float64], state: TheilSenRegressorState) -> NDArray[np.float64]:
    """Predict dense Theil-Sen regression outputs."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    return np.asarray(np.dot(checked_x, state.coef) + state.intercept, dtype=np.float64)


def _lars_path_state_valid(state: LarsPathState) -> bool:
    return bool(
        state.alphas.ndim == 1
        and state.active.ndim == 1
        and state.coefs.ndim == 2
        and state.coefs.shape[0] == state.n_features_in
        and state.coefs.shape[1] == state.alphas.shape[0]
        and state.active.shape[0] <= state.n_iter
        and state.n_iter >= 0
        and state.method == "lar"
        and state.alpha_min >= 0.0
        and state.n_samples >= 1
        and state.n_features_in >= 1
        and np.all(np.isfinite(state.alphas))
        and np.all(state.alphas >= 0.0)
        and np.all(np.isfinite(state.coefs))
        and np.all((state.active >= 0) & (state.active < state.n_features_in))
    )


def _lars_state_valid(state: LarsState) -> bool:
    return bool(
        state.coef.shape == (state.n_features_in,)
        and state.coef_path.shape[0] == state.n_features_in
        and state.coef_path.shape[1] == state.alphas.shape[0]
        and state.active.ndim == 1
        and state.n_iter >= 0
        and isinstance(state.fit_intercept, bool)
        and state.n_nonzero_coefs >= 1
        and state.n_features_in >= 1
        and np.isfinite(state.intercept)
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.alphas))
        and np.all(np.isfinite(state.coef_path))
        and np.all((state.active >= 0) & (state.active < state.n_features_in))
    )


def _lars_feature_count_matches(X: NDArray[np.float64], state: LarsState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _lars_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isfinite(values)))


def _lars_precompute_valid(precompute: bool | str | NDArray[np.float64]) -> bool:
    return bool(isinstance(precompute, bool) or (isinstance(precompute, str) and precompute == "auto") or _square_matrix(precompute))


def _lars_path_output_from_gram(
    Xy: NDArray[np.float64],
    Gram: NDArray[np.float64],
    *,
    n_samples: int,
    max_iter: int,
    alpha_min: float,
) -> LarsPathState:
    n_features = Xy.shape[0]
    max_features = min(max_iter, n_features)
    active: list[int] = []
    sign_active: list[float] = []
    coef = np.zeros(n_features, dtype=np.float64)
    alphas = [float(np.max(np.abs(Xy)) / n_samples)]
    coefs = [coef.copy()]
    cov = Xy.copy()
    tiny = np.finfo(np.float64).tiny
    equality_tolerance = np.finfo(np.float32).eps

    n_iter = 0
    while n_iter < max_features and len(active) < n_features:
        inactive = np.asarray([idx for idx in range(n_features) if idx not in active], dtype=np.int64)
        if inactive.size == 0:
            break
        current_abs = np.abs(cov[inactive])
        selected = int(inactive[int(np.argmax(current_abs))])
        C = float(abs(cov[selected]))
        if C / n_samples <= alpha_min + equality_tolerance or C <= tiny:
            if alphas[-1] != alpha_min and n_iter > 0:
                previous_alpha = alphas[-1]
                if abs(previous_alpha - C / n_samples) > equality_tolerance:
                    ss = (previous_alpha - alpha_min) / (previous_alpha - C / n_samples)
                    coefs[-1] = coefs[-2] + ss * (coefs[-1] - coefs[-2])
                alphas[-1] = float(alpha_min)
            break
        active.append(selected)
        selected_sign = float(np.sign(cov[selected]))
        sign_active.append(1.0 if selected_sign == 0.0 else selected_sign)
        active_index = np.asarray(active, dtype=np.int64)
        signs = np.asarray(sign_active, dtype=np.float64)
        gram_active = Gram[np.ix_(active_index, active_index)]
        try:
            least_squares = linalg.solve(gram_active, signs, assume_a="sym")
        except linalg.LinAlgError:
            least_squares, _, _, _ = linalg.lstsq(gram_active, signs, cond=None)
        denom = float(np.sum(least_squares * signs))
        if denom <= tiny:
            break
        AA = float(1.0 / np.sqrt(denom))
        direction = AA * least_squares
        corr_eq_dir = Gram[:, active_index] @ direction
        inactive_after = np.asarray([idx for idx in range(n_features) if idx not in active], dtype=np.int64)
        if inactive_after.size:
            g1 = _min_positive((C - cov[inactive_after]) / (AA - corr_eq_dir[inactive_after] + tiny))
            g2 = _min_positive((C + cov[inactive_after]) / (AA + corr_eq_dir[inactive_after] + tiny))
            gamma = min(g1, g2, C / AA)
        else:
            gamma = C / AA
        if not np.isfinite(gamma):
            gamma = C / AA
        coef[active_index] += gamma * direction
        cov = Xy - Gram @ coef
        n_iter += 1
        next_alpha = 0.0 if inactive_after.size == 0 else float(np.max(np.abs(cov[inactive_after])) / n_samples)
        coefs.append(coef.copy())
        alphas.append(max(next_alpha, 0.0))
        if next_alpha <= alpha_min + equality_tolerance:
            if abs(next_alpha - alpha_min) > equality_tolerance and len(coefs) >= 2:
                previous_alpha = alphas[-2]
                if abs(previous_alpha - next_alpha) > equality_tolerance:
                    ss = (previous_alpha - alpha_min) / (previous_alpha - next_alpha)
                    coefs[-1] = coefs[-2] + ss * (coefs[-1] - coefs[-2])
            alphas[-1] = float(alpha_min)
            break

    return LarsPathState(
        alphas=np.asarray(alphas, dtype=np.float64),
        active=np.asarray(active, dtype=np.int64),
        coefs=np.asarray(coefs, dtype=np.float64).T,
        n_iter=int(n_iter),
        method="lar",
        alpha_min=float(alpha_min),
        n_samples=int(n_samples),
        n_features_in=int(n_features),
    )


def _min_positive(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return float("inf")
    return float(np.min(finite))


@register_atom(witness_lars_path)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda Xy, X: Xy is None or (np.asarray(Xy).ndim == 1 and np.asarray(Xy).shape[0] == np.asarray(X).shape[1] and np.all(np.isfinite(np.asarray(Xy)))), "Xy must be 1D and match feature count")
@icontract.require(lambda Gram: Gram is None or isinstance(Gram, (bool, str)) or _square_matrix(Gram), "Gram must be None, boolean, auto, or square")
@icontract.require(lambda max_iter: isinstance(max_iter, int) and not isinstance(max_iter, bool) and max_iter >= 1, "max_iter must be positive")
@icontract.require(lambda alpha_min: _nonnegative_finite(alpha_min), "alpha_min must be non-negative")
@icontract.require(lambda method: method == "lar", "only method='lar' is covered")
@icontract.require(lambda copy_X: _bool_value(copy_X), "copy_X must be boolean")
@icontract.require(lambda eps: _positive_finite(eps), "eps must be positive")
@icontract.require(lambda copy_Gram: _bool_value(copy_Gram), "copy_Gram must be boolean")
@icontract.require(lambda verbose: verbose in {False, 0}, "verbose output is outside this atom scope")
@icontract.require(lambda return_path: return_path is True, "return_path=False is outside this atom scope")
@icontract.require(lambda return_n_iter: _bool_value(return_n_iter), "return_n_iter must be boolean")
@icontract.require(lambda positive: positive is False, "positive=True is outside this LARS atom scope")
@icontract.ensure(lambda result: _lars_path_state_valid(result), "LARS path state must contain finite path arrays")
def lars_path(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    Xy: NDArray[np.float64] | None = None,
    *,
    Gram: NDArray[np.float64] | bool | str | None = None,
    max_iter: int = 500,
    alpha_min: float = 0.0,
    method: str = "lar",
    copy_X: bool = True,
    eps: float = np.finfo(float).eps,
    copy_Gram: bool = True,
    verbose: int | bool = 0,
    return_path: bool = True,
    return_n_iter: bool = False,
    positive: bool = False,
) -> LarsPathState:
    """Compute a dense unconstrained LARS coefficient path."""
    del copy_X, eps, copy_Gram, verbose, return_path, return_n_iter, positive
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    xy = np.asarray(checked_x.T @ checked_y if Xy is None else Xy, dtype=np.float64)
    if Gram is None or (isinstance(Gram, bool) and Gram is False):
        gram = np.asarray(checked_x.T @ checked_x, dtype=np.float64)
    elif (isinstance(Gram, bool) and Gram is True) or (isinstance(Gram, str) and Gram == "auto"):
        gram = np.asarray(checked_x.T @ checked_x, dtype=np.float64)
    else:
        gram = check_array(Gram, dtype=np.float64, ensure_2d=True)
    return _lars_path_output_from_gram(xy, gram, n_samples=checked_x.shape[0], max_iter=max_iter, alpha_min=alpha_min)


@register_atom(witness_lars_path_gram)
@icontract.require(lambda Xy: np.asarray(Xy).ndim == 1, "Xy must be 1D")
@icontract.require(lambda Gram: _square_matrix(Gram), "Gram must be square")
@icontract.require(lambda Gram, Xy: _gram_and_xy_match(Gram, Xy), "Xy must match Gram feature count")
@icontract.require(lambda Gram, Xy: _finite_gram_inputs(Gram, Xy), "Gram and Xy must contain finite numeric values")
@icontract.require(lambda n_samples: isinstance(n_samples, int) and not isinstance(n_samples, bool) and n_samples >= 1, "n_samples must be positive")
@icontract.require(lambda max_iter: isinstance(max_iter, int) and not isinstance(max_iter, bool) and max_iter >= 1, "max_iter must be positive")
@icontract.require(lambda alpha_min: _nonnegative_finite(alpha_min), "alpha_min must be non-negative")
@icontract.require(lambda method: method == "lar", "only method='lar' is covered")
@icontract.require(lambda copy_X: _bool_value(copy_X), "copy_X must be boolean")
@icontract.require(lambda eps: _positive_finite(eps), "eps must be positive")
@icontract.require(lambda copy_Gram: _bool_value(copy_Gram), "copy_Gram must be boolean")
@icontract.require(lambda verbose: verbose in {False, 0}, "verbose output is outside this atom scope")
@icontract.require(lambda return_path: return_path is True, "return_path=False is outside this atom scope")
@icontract.require(lambda return_n_iter: _bool_value(return_n_iter), "return_n_iter must be boolean")
@icontract.require(lambda positive: positive is False, "positive=True is outside this LARS atom scope")
@icontract.ensure(lambda result: _lars_path_state_valid(result), "LARS path state must contain finite path arrays")
def lars_path_gram(
    Xy: NDArray[np.float64],
    Gram: NDArray[np.float64],
    *,
    n_samples: int,
    max_iter: int = 500,
    alpha_min: float = 0.0,
    method: str = "lar",
    copy_X: bool = True,
    eps: float = np.finfo(float).eps,
    copy_Gram: bool = True,
    verbose: int | bool = 0,
    return_path: bool = True,
    return_n_iter: bool = False,
    positive: bool = False,
) -> LarsPathState:
    """Compute an unconstrained LARS path from sufficient statistics."""
    del copy_X, eps, copy_Gram, verbose, return_path, return_n_iter, positive
    checked_xy = np.asarray(Xy, dtype=np.float64)
    checked_gram = check_array(Gram, dtype=np.float64, ensure_2d=True)
    return _lars_path_output_from_gram(checked_xy, checked_gram, n_samples=n_samples, max_iter=max_iter, alpha_min=alpha_min)


@register_atom(witness_lars_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must contain finite numeric values")
@icontract.require(lambda Xy, X: Xy is None or (np.asarray(Xy).ndim == 1 and np.asarray(Xy).shape[0] == np.asarray(X).shape[1] and np.all(np.isfinite(np.asarray(Xy)))), "Xy must be 1D and match feature count")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda verbose: verbose in {False, 0}, "verbose output is outside this atom scope")
@icontract.require(lambda precompute: _lars_precompute_valid(precompute), "precompute must be boolean, auto, or square Gram")
@icontract.require(lambda n_nonzero_coefs: isinstance(n_nonzero_coefs, int) and not isinstance(n_nonzero_coefs, bool) and n_nonzero_coefs >= 1, "n_nonzero_coefs must be positive")
@icontract.require(lambda eps: _positive_finite(eps), "eps must be positive")
@icontract.require(lambda copy_X: _bool_value(copy_X), "copy_X must be boolean")
@icontract.require(lambda fit_path: fit_path is True, "fit_path=False is outside this atom scope")
@icontract.require(lambda jitter: jitter is None, "jitter is outside this atom scope")
@icontract.require(lambda random_state: random_state is None, "random_state is only used with jitter")
@icontract.ensure(lambda result: _lars_state_valid(result), "LARS state must contain finite fitted coefficients")
def lars_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    Xy: NDArray[np.float64] | None = None,
    *,
    fit_intercept: bool = True,
    verbose: int | bool = False,
    precompute: bool | str | NDArray[np.float64] = "auto",
    n_nonzero_coefs: int = 500,
    eps: float = np.finfo(float).eps,
    copy_X: bool = True,
    fit_path: bool = True,
    jitter: None = None,
    random_state: None = None,
) -> LarsState:
    """Fit dense single-output least-angle-regression coefficients."""
    del verbose, eps, copy_X, fit_path, jitter, random_state
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    checked_y = check_array(y, dtype=np.float64, ensure_2d=False, input_name="y")
    centered_x, centered_y_2d, x_offset, y_offset = _center_and_rescale(checked_x, checked_y.reshape(-1, 1), fit_intercept, None)
    centered_y = np.ravel(centered_y_2d)
    xy = None if Xy is None else np.asarray(Xy, dtype=np.float64)
    if isinstance(precompute, np.ndarray):
        gram: NDArray[np.float64] | bool | str | None = np.asarray(precompute, dtype=np.float64)
    elif precompute is True or (precompute == "auto" and checked_x.shape[0] > checked_x.shape[1]):
        gram = np.asarray(centered_x.T @ centered_x, dtype=np.float64)
    else:
        gram = None
    path = lars_path(centered_x, centered_y, xy, Gram=gram, max_iter=n_nonzero_coefs)
    coef = np.asarray(path.coefs[:, -1], dtype=np.float64)
    intercept = float(y_offset[0] - np.dot(x_offset, coef)) if fit_intercept else 0.0
    return LarsState(
        coef=coef,
        intercept=intercept,
        alphas=path.alphas,
        active=path.active,
        coef_path=path.coefs,
        n_iter=path.n_iter,
        fit_intercept=fit_intercept,
        n_nonzero_coefs=int(n_nonzero_coefs),
        n_features_in=int(checked_x.shape[1]),
    )


@register_atom(witness_lars_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _lars_feature_count_matches(X, state), "X feature count must match fitted LARS state")
@icontract.require(lambda state: _lars_state_valid(state), "state must be a fitted LARS state")
@icontract.ensure(lambda result, X: _lars_prediction_valid(result, X), "predictions must be finite per-row values")
def lars_predict(X: NDArray[np.float64], state: LarsState) -> NDArray[np.float64]:
    """Predict dense LARS regression outputs."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    return np.asarray(np.dot(checked_x, state.coef) + state.intercept, dtype=np.float64)
