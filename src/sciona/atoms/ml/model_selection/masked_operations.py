"""Row-aligned masked fitting and calibration for composable training graphs."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom
from sciona.atoms.ml.xgboost.model_io import (fit_regression_model, fit_binary_model,
    witness_fit_regression_model, witness_fit_binary_model)
from sciona.atoms.ml.calibration.conditional_residuals import (
    estimate_conditional_offsets, witness_estimate_conditional_offsets)


def _mask(mask, rows, nonempty=True):
    value = np.asarray(mask)
    if value.dtype != np.bool_ or value.shape != (rows,) or (nonempty and not value.any()):
        raise ValueError('Aligned boolean row mask with a sufficient selection required')
    return value


def _mask_metadata(mask, rows):
    if mask.dtype != 'bool' or mask.shape != (rows,):
        raise ValueError('Aligned boolean mask metadata required')
    if mask.dim is not None and not mask.dim.is_dimensionless:
        raise ValueError('Selection masks are dimensionless')


def witness_intersect_masks(left: AbstractArray, right: AbstractArray) -> AbstractArray:
    if len(left.shape) != 1:
        raise ValueError('Vector masks required')
    _mask_metadata(left, left.shape[0])
    _mask_metadata(right, left.shape[0])
    return AbstractArray(shape=left.shape, dtype='bool')


@register_atom(witness_intersect_masks)
def intersect_masks(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Intersect aligned row selections; an empty intersection remains valid."""
    a = np.asarray(left)
    if a.ndim != 1:
        raise ValueError('Vector masks required')
    return _mask(a, len(a), False) & _mask(right, len(a), False)


def witness_fit_masked_regression(features: AbstractArray, targets: AbstractArray, feature_names: list, selection: AbstractArray) -> dict:
    state = witness_fit_regression_model(features, targets, feature_names)
    _mask_metadata(selection, features.shape[0])
    return state


def witness_fit_masked_binary(features: AbstractArray, targets: AbstractArray, feature_names: list, selection: AbstractArray) -> dict:
    state = witness_fit_binary_model(features, targets, feature_names)
    _mask_metadata(selection, features.shape[0])
    return state


def _selected(features, targets, selection):
    x, y = np.asarray(features), np.asarray(targets)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError('Aligned feature matrix and target vector required')
    mask = _mask(selection, len(x))
    return x[mask], y[mask]


@register_atom(witness_fit_masked_regression)
def fit_masked_regression(features: NDArray[np.float64], targets: NDArray[np.float64], feature_names: list, selection: NDArray[np.bool_]) -> dict:
    """Fit the reusable one-thread regressor on selected rows in original order.

    Selection is explicit, never inferred from missing values. Only selected
    values enter fitting; native model and ordered feature contracts are retained.
    The graph retains full row-aligned arrays and a mask instead of requiring
    symbolic inference of a data-dependent selected row count.
    """
    x, y = _selected(features, targets, selection)
    return fit_regression_model(x, y, feature_names)


@register_atom(witness_fit_masked_binary)
def fit_masked_binary(features: NDArray[np.float64], targets: NDArray[np.int64], feature_names: list, selection: NDArray[np.bool_]) -> dict:
    """Fit the reusable one-thread binary model; selected rows need both classes."""
    x, y = _selected(features, targets, selection)
    return fit_binary_model(x, y, feature_names)


def witness_estimate_masked_offsets(observed: AbstractArray, calibration_predictions: AbstractArray,
        calibration_probabilities: AbstractArray, threshold: AbstractScalar, selection: AbstractArray) -> AbstractArray:
    output = witness_estimate_conditional_offsets(observed, calibration_predictions, calibration_probabilities, threshold)
    _mask_metadata(selection, observed.shape[0])
    return output


@register_atom(witness_estimate_masked_offsets)
def estimate_masked_offsets(observed: NDArray[np.float64], calibration_predictions: NDArray[np.float64],
        calibration_probabilities: NDArray[np.float64], threshold: float, selection: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Estimate signed conditional median offsets only on selected calibration rows.

    Both strict probability groups must exist inside the selection. Row order,
    target units and the existing estimator's finite float64 contract are retained.
    """
    arrays = [np.asarray(v) for v in (observed, calibration_predictions, calibration_probabilities)]
    if arrays[0].ndim != 1 or any(v.shape != arrays[0].shape for v in arrays):
        raise ValueError('Aligned calibration vectors required')
    mask = _mask(selection, len(arrays[0]))
    return estimate_conditional_offsets(*(v[mask] for v in arrays), threshold)
