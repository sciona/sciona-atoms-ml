"""Ghost witnesses for selected sklearn calibration atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import CalibratedClassifierCVState, SigmoidCalibrationState, TemperatureScalingState


def witness_calibration_curve(
    y_true: AbstractArray,
    y_prob: AbstractArray,
    *,
    pos_label: int | float | bool | str | None = None,
    n_bins: int = 5,
    strategy: str = "uniform",
) -> tuple[AbstractArray, AbstractArray]:
    """Describe true and predicted probability vectors from calibration bins."""
    del pos_label
    if len(y_true.shape) != 1:
        raise ValueError("y_true must be 1D")
    if len(y_prob.shape) != 1:
        raise ValueError("y_prob must be 1D")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have equal sample count")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    prob_true = AbstractArray(shape=(n_bins,), dtype="float64", min_val=0.0, max_val=1.0)
    prob_pred = AbstractArray(shape=(n_bins,), dtype="float64", min_val=0.0, max_val=1.0)
    return prob_true, prob_pred


def witness_sigmoid_calibration_fit(
    predictions: AbstractArray,
    y: AbstractArray,
    *,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe fitting Platt sigmoid calibration parameters."""
    del sample_weight
    if len(predictions.shape) != 1:
        raise ValueError("predictions must be 1D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if predictions.shape[0] != y.shape[0]:
        raise ValueError("predictions and y must have equal sample count")
    return AbstractArray(shape=(2,), dtype="float64")


def witness_sigmoid_calibration_predict(
    predictions: AbstractArray,
    state: SigmoidCalibrationState,
) -> AbstractArray:
    """Describe calibrated probabilities from sigmoid parameters."""
    del state
    if len(predictions.shape) != 1:
        raise ValueError("predictions must be 1D")
    return AbstractArray(shape=(int(predictions.shape[0]),), dtype="float64", min_val=0.0, max_val=1.0)


def witness_temperature_scaling_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe fitting an inverse temperature scaling parameter."""
    del sample_weight
    if len(X.shape) not in {1, 2}:
        raise ValueError("X must be 1D or 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have equal sample count")
    return AbstractArray(shape=(1,), dtype="float64", min_val=0.0)


def witness_temperature_scaling_predict(
    X: AbstractArray,
    state: TemperatureScalingState,
) -> AbstractArray:
    """Describe temperature-scaled class probabilities."""
    del state
    if len(X.shape) == 1:
        return AbstractArray(shape=(int(X.shape[0]), 2), dtype="float64", min_val=0.0, max_val=1.0)
    if len(X.shape) == 2:
        width = 2 if int(X.shape[1]) == 1 else int(X.shape[1])
        return AbstractArray(shape=(int(X.shape[0]), width), dtype="float64", min_val=0.0, max_val=1.0)
    raise ValueError("X must be 1D or 2D")


def witness_calibrated_classifier_cv_fit(
    estimator: object,
    X: AbstractArray,
    y: AbstractArray,
    *,
    method: str = "sigmoid",
    cv: int | None = None,
    n_jobs: int | None = None,
    ensemble: bool | str = "auto",
) -> AbstractArray:
    """Describe fitting a calibrated classifier CV meta-estimator."""
    del estimator, cv, n_jobs, ensemble
    n_samples, _ = _check_2d(X)
    if len(y.shape) != 1 or int(y.shape[0]) != n_samples:
        raise ValueError("X and y must have equal sample count")
    if method not in {"sigmoid", "isotonic", "temperature"}:
        raise ValueError("invalid calibration method")
    return AbstractArray(shape=(n_samples,), dtype="object")


def witness_calibrated_classifier_cv_predict_proba(
    X: AbstractArray,
    state: CalibratedClassifierCVState,
) -> AbstractArray:
    """Describe probabilities from a fitted calibrated classifier."""
    n_samples, n_features = _check_2d(X)
    if state.n_features_in is not None and n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, int(state.classes.shape[0])), dtype="float64", min_val=0.0, max_val=1.0)


def witness_calibrated_classifier_cv_predict(
    X: AbstractArray,
    state: CalibratedClassifierCVState,
) -> AbstractArray:
    """Describe class predictions from calibrated probabilities."""
    n_samples, n_features = _check_2d(X)
    if state.n_features_in is not None and n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples,), dtype="object")


def _check_2d(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return int(X.shape[0]), int(X.shape[1])
