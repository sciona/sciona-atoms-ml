"""Selected calibration atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, softmax
from sklearn.calibration import CalibratedClassifierCV, _sigmoid_calibration
from sklearn.calibration import _TemperatureScaling as SklearnTemperatureScaling
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.utils.validation import _check_pos_label_consistency

from sciona.ghost.registry import register_atom

from .state_models import CalibratedClassifierCVState, SigmoidCalibrationState, TemperatureScalingState
from .witnesses import (
    witness_calibrated_classifier_cv_fit,
    witness_calibrated_classifier_cv_predict,
    witness_calibrated_classifier_cv_predict_proba,
    witness_calibration_curve,
    witness_sigmoid_calibration_fit,
    witness_sigmoid_calibration_predict,
    witness_temperature_scaling_fit,
    witness_temperature_scaling_predict,
)

CurveResult = tuple[NDArray[np.float64], NDArray[np.float64]]
ArrayLike = NDArray[np.float64] | list[float] | list[list[float]]


def _is_1d(x: NDArray[np.float64]) -> bool:
    return bool(np.asarray(x).ndim == 1)


def _same_length(y_true: NDArray[np.float64], y_prob: NDArray[np.float64]) -> bool:
    return int(np.asarray(y_true).shape[0]) == int(np.asarray(y_prob).shape[0])


def _valid_strategy(strategy: str) -> bool:
    return strategy in {"uniform", "quantile"}


def _probabilities_in_unit_interval(y_prob: NDArray[np.float64]) -> bool:
    values = np.asarray(y_prob)
    return bool(values.size > 0 and values.min() >= 0.0 and values.max() <= 1.0)


def _curve_result_valid(result: CurveResult, n_bins: int) -> bool:
    prob_true, prob_pred = result
    if prob_true.shape != prob_pred.shape or prob_true.ndim != 1:
        return False
    if prob_true.shape[0] > n_bins:
        return False
    return bool(
        np.all((prob_true >= 0.0) & (prob_true <= 1.0))
        and np.all((prob_pred >= 0.0) & (prob_pred <= 1.0))
    )


def _sigmoid_state_valid(state: SigmoidCalibrationState) -> bool:
    return bool(np.isfinite(state.a) and np.isfinite(state.b))


def _temperature_state_valid(state: TemperatureScalingState) -> bool:
    return bool(np.isfinite(state.beta) and state.beta > 0.0)


def _is_1d_or_2d(x: ArrayLike) -> bool:
    return bool(np.asarray(x).ndim in {1, 2})


def _same_first_dim(X: ArrayLike, y: NDArray[np.float64]) -> bool:
    return int(np.asarray(X).shape[0]) == int(np.asarray(y).shape[0])


def _calibration_method_valid(method: str) -> bool:
    return method in {"sigmoid", "isotonic", "temperature"}


def _ensemble_valid(ensemble: bool | str) -> bool:
    return isinstance(ensemble, bool) or ensemble == "auto"


def _cv_valid(cv: int | None) -> bool:
    return cv is None or cv >= 2


def _calibrated_state_valid(state: CalibratedClassifierCVState) -> bool:
    return bool(
        hasattr(state.estimator, "predict_proba")
        and state.classes.ndim == 1
        and state.classes.shape[0] >= 2
        and _calibration_method_valid(state.method)
        and _ensemble_valid(state.ensemble)
    )


def _probability_rows_valid(result: NDArray[np.float64]) -> bool:
    row_sums = np.sum(result, axis=1)
    return bool(np.all(np.isfinite(result)) and np.all(result >= 0.0) and np.allclose(row_sums, np.ones_like(row_sums)))


def _calibrated_proba_valid(result: NDArray[np.float64], X: ArrayLike, state: CalibratedClassifierCVState) -> bool:
    return bool(result.shape == (np.asarray(X).shape[0], state.classes.shape[0]) and _probability_rows_valid(result))


def _calibrated_predict_valid(result: NDArray[np.object_], X: ArrayLike, state: CalibratedClassifierCVState) -> bool:
    return bool(result.shape == (np.asarray(X).shape[0],) and np.isin(result, state.classes).all())


def _convert_to_logits_np(X: ArrayLike, eps: float = 1e-12) -> NDArray[np.float64]:
    values = np.asarray(check_array(X, dtype=[np.float64, np.float32], ensure_2d=False), dtype=np.float64)
    if values.ndim == 2 and values.shape[1] > 1:
        if np.all((values >= 0.0) & (values <= 1.0)) and np.allclose(np.sum(values, axis=1), 1.0):
            return np.log(values + eps)
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return np.column_stack([-values[:, 0], values[:, 0]])
    if values.ndim == 1:
        return np.column_stack([-values, values])
    raise ValueError("X must be 1D or 2D")


@register_atom(witness_calibration_curve)
@icontract.require(lambda y_true: _is_1d(y_true), "y_true must be a 1D vector")
@icontract.require(lambda y_prob: _is_1d(y_prob), "y_prob must be a 1D vector")
@icontract.require(lambda y_true, y_prob: _same_length(y_true, y_prob), "y_true and y_prob must have equal sample count")
@icontract.require(lambda y_prob: _probabilities_in_unit_interval(y_prob), "y_prob values must be in [0, 1]")
@icontract.require(lambda n_bins: n_bins >= 1, "n_bins must be at least 1")
@icontract.require(lambda strategy: _valid_strategy(strategy), "strategy must be 'uniform' or 'quantile'")
@icontract.ensure(lambda result, n_bins: _curve_result_valid(result, n_bins), "curve outputs must be probability vectors with at most n_bins entries")
def calibration_curve(
    y_true: NDArray[np.float64],
    y_prob: NDArray[np.float64],
    *,
    pos_label: int | float | bool | str | None = None,
    n_bins: int = 5,
    strategy: str = "uniform",
) -> CurveResult:
    """Compute positive rate and mean predicted probability by calibration bin."""
    checked_y_true = column_or_1d(y_true)
    checked_y_prob = column_or_1d(y_prob)
    check_consistent_length(checked_y_true, checked_y_prob)
    checked_pos_label = _check_pos_label_consistency(pos_label, checked_y_true)

    if checked_y_prob.min() < 0 or checked_y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(checked_y_true)
    if len(labels) > 2:
        raise ValueError(f"Only binary classification is supported. Provided labels {labels}.")
    positive_mask = checked_y_true == checked_pos_label

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(checked_y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy must be either 'quantile' or 'uniform'.")

    binids = np.searchsorted(bins[1:-1], checked_y_prob)
    bin_sums = np.bincount(binids, weights=checked_y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=positive_mask, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    return np.asarray(prob_true, dtype=np.float64), np.asarray(prob_pred, dtype=np.float64)


@register_atom(witness_sigmoid_calibration_fit)
@icontract.require(lambda predictions: _is_1d(predictions), "predictions must be a 1D vector")
@icontract.require(lambda y: _is_1d(y), "y must be a 1D vector")
@icontract.require(lambda predictions, y: _same_length(predictions, y), "predictions and y must have equal sample count")
@icontract.ensure(lambda result: _sigmoid_state_valid(result), "sigmoid calibration parameters must be finite")
def sigmoid_calibration_fit(
    predictions: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    sample_weight: NDArray[np.float64] | None = None,
) -> SigmoidCalibrationState:
    """Fit Platt sigmoid calibration slope and intercept."""
    a, b = _sigmoid_calibration(predictions, y, sample_weight=sample_weight)
    return SigmoidCalibrationState(a=float(a), b=float(b))


@register_atom(witness_sigmoid_calibration_predict)
@icontract.require(lambda predictions: _is_1d(predictions), "predictions must be a 1D vector")
@icontract.require(lambda state: _sigmoid_state_valid(state), "sigmoid calibration parameters must be finite")
@icontract.ensure(lambda result, predictions: result.shape == np.asarray(predictions).shape, "calibrated probabilities must match input shape")
@icontract.ensure(lambda result: bool(np.all((result >= 0.0) & (result <= 1.0))), "calibrated probabilities must be in [0, 1]")
def sigmoid_calibration_predict(
    predictions: NDArray[np.float64],
    state: SigmoidCalibrationState,
) -> NDArray[np.float64]:
    """Apply fitted Platt sigmoid calibration to raw predictions."""
    checked = column_or_1d(predictions)
    return np.asarray(expit(-(state.a * checked + state.b)), dtype=np.float64)


@register_atom(witness_temperature_scaling_fit)
@icontract.require(lambda X: _is_1d_or_2d(X), "X must be a 1D or 2D score array")
@icontract.require(lambda y: _is_1d(y), "y must be a 1D label vector")
@icontract.require(lambda X, y: _same_first_dim(X, y), "X and y must have equal sample count")
@icontract.ensure(lambda result: _temperature_state_valid(result), "temperature scaling beta must be positive")
def temperature_scaling_fit(
    X: ArrayLike,
    y: NDArray[np.float64],
    *,
    sample_weight: NDArray[np.float64] | None = None,
) -> TemperatureScalingState:
    """Fit the inverse temperature for softmax probability calibration."""
    calibrator = SklearnTemperatureScaling().fit(X, y, sample_weight=sample_weight)
    return TemperatureScalingState(beta=float(calibrator.beta_))


@register_atom(witness_temperature_scaling_predict)
@icontract.require(lambda X: _is_1d_or_2d(X), "X must be a 1D or 2D score array")
@icontract.require(lambda state: _temperature_state_valid(state), "temperature scaling beta must be positive")
@icontract.ensure(lambda result: _probability_rows_valid(result), "temperature-scaled rows must be probabilities")
def temperature_scaling_predict(
    X: ArrayLike,
    state: TemperatureScalingState,
) -> NDArray[np.float64]:
    """Apply fitted inverse temperature scaling to logits or probabilities."""
    logits = _convert_to_logits_np(X)
    return np.asarray(softmax(state.beta * logits, axis=1), dtype=np.float64)


@register_atom(witness_calibrated_classifier_cv_fit)
@icontract.require(lambda estimator: estimator is None or hasattr(estimator, "fit"), "estimator must implement fit or be None")
@icontract.require(lambda X: bool(np.asarray(X).ndim == 2), "X must be a 2D feature matrix")
@icontract.require(lambda y: _is_1d(y), "y must be a 1D label vector")
@icontract.require(lambda X, y: _same_first_dim(X, y), "X and y must have equal sample count")
@icontract.require(lambda method: _calibration_method_valid(method), "invalid calibration method")
@icontract.require(lambda cv: _cv_valid(cv), "cv must be at least two or None")
@icontract.require(lambda ensemble: _ensemble_valid(ensemble), "ensemble must be boolean or 'auto'")
@icontract.ensure(lambda result: _calibrated_state_valid(result), "calibrated classifier state must be fitted")
def calibrated_classifier_cv_fit(
    estimator: object | None,
    X: ArrayLike,
    y: NDArray[np.float64],
    *,
    method: str = "sigmoid",
    cv: int | None = None,
    n_jobs: int | None = None,
    ensemble: bool | str = "auto",
) -> CalibratedClassifierCVState:
    """Fit a calibrated classifier CV meta-estimator and return immutable state."""
    fitted = CalibratedClassifierCV(estimator=estimator, method=method, cv=cv, n_jobs=n_jobs, ensemble=ensemble).fit(X, y)
    n_features = getattr(fitted, "n_features_in_", None)
    return CalibratedClassifierCVState(
        estimator=fitted,
        classes=np.asarray(fitted.classes_, dtype=object),
        method=method,
        ensemble=ensemble,
        n_features_in=None if n_features is None else int(n_features),
    )


@register_atom(witness_calibrated_classifier_cv_predict_proba)
@icontract.require(lambda X: bool(np.asarray(X).ndim == 2), "X must be a 2D feature matrix")
@icontract.require(lambda state: _calibrated_state_valid(state), "calibrated classifier state must be fitted")
@icontract.ensure(lambda result, X, state: _calibrated_proba_valid(result, X, state), "calibrated probabilities must match classes and sum to one")
def calibrated_classifier_cv_predict_proba(
    X: ArrayLike,
    state: CalibratedClassifierCVState,
) -> NDArray[np.float64]:
    """Predict calibrated class probabilities from fitted CV state."""
    return np.asarray(state.estimator.predict_proba(X), dtype=np.float64)


@register_atom(witness_calibrated_classifier_cv_predict)
@icontract.require(lambda X: bool(np.asarray(X).ndim == 2), "X must be a 2D feature matrix")
@icontract.require(lambda state: _calibrated_state_valid(state), "calibrated classifier state must be fitted")
@icontract.ensure(lambda result, X, state: _calibrated_predict_valid(result, X, state), "predicted labels must be fitted classes")
def calibrated_classifier_cv_predict(
    X: ArrayLike,
    state: CalibratedClassifierCVState,
) -> NDArray[np.object_]:
    """Predict labels from fitted calibrated probabilities."""
    return np.asarray(state.estimator.predict(X), dtype=object)
