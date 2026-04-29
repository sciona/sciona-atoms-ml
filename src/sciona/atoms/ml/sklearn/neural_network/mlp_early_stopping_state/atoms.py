"""MLP early-stopping state helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from sciona.atoms.ml.sklearn.preprocessing.atoms import label_binarizer_inverse_transform
from sciona.atoms.ml.sklearn.preprocessing.state_models import LabelBinarizerState

from .witnesses import (
    witness_mlp_monitor_best_state,
    witness_mlp_restore_best_parameters,
    witness_mlp_stochastic_validation_targets,
    witness_mlp_validation_scores_append,
)

WeightArray = NDArray[np.float64]
ParameterTuple = tuple[WeightArray, ...]
DecodedTargets = NDArray[np.object_] | NDArray[np.float64] | sp.csr_matrix
ValidationTargets = NDArray[np.float64] | NDArray[np.bool_]


def _bool_valid(value: object) -> bool:
    return isinstance(value, bool)


def _finite_scalar(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def _finite_or_neg_inf_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (np.isfinite(float(value)) or float(value) == float(-np.inf))
    )


def _target_matrix_valid(values: object) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in {1, 2}
        and array.shape[0] >= 1
        and (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        )
        and np.all(np.isfinite(array.astype(np.float64, copy=False)))
    )


def _label_binarizer_state_valid(state: LabelBinarizerState) -> bool:
    classes = np.asarray(state.classes)
    return bool(
        classes.ndim == 1
        and classes.shape[0] >= 1
        and isinstance(state.y_type, str)
        and isinstance(state.sparse_input, bool)
        and isinstance(state.neg_label, int)
        and isinstance(state.pos_label, int)
        and state.neg_label < state.pos_label
        and isinstance(state.sparse_output, bool)
    )


def _validation_scores_valid(values: object) -> bool:
    if not isinstance(values, tuple):
        return False
    return all(_finite_scalar(value) for value in values)


def _weight_array_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim in {1, 2} and array.size >= 1 and np.all(np.isfinite(array)))


def _parameter_sequence_valid(values: object) -> bool:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False
    arrays = list(values)
    return bool(arrays and all(_weight_array_valid(array) for array in arrays))


def _parameter_shapes_match(left: Sequence[WeightArray], right: Sequence[WeightArray]) -> bool:
    return bool(
        len(left) == len(right)
        and all(
            np.asarray(l, dtype=np.float64).shape == np.asarray(r, dtype=np.float64).shape
            for l, r in zip(left, right)
        )
    )


def _parameter_state_inputs_valid(
    best_coefs: Sequence[WeightArray],
    best_intercepts: Sequence[WeightArray],
    coefs: Sequence[WeightArray],
    intercepts: Sequence[WeightArray],
) -> bool:
    return bool(
        _parameter_sequence_valid(best_coefs)
        and _parameter_sequence_valid(best_intercepts)
        and _parameter_sequence_valid(coefs)
        and _parameter_sequence_valid(intercepts)
        and len(best_coefs) == len(best_intercepts) == len(coefs) == len(intercepts)
        and _parameter_shapes_match(best_coefs, coefs)
        and _parameter_shapes_match(best_intercepts, intercepts)
    )


def _parameter_tuple_valid(values: object) -> bool:
    return isinstance(values, tuple) and _parameter_sequence_valid(values)


def _restored_parameters_valid(result: object, best_coefs: Sequence[WeightArray], best_intercepts: Sequence[WeightArray]) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2):
        return False
    restored_coefs, restored_intercepts = result
    if not (_parameter_tuple_valid(restored_coefs) and _parameter_tuple_valid(restored_intercepts)):
        return False
    return bool(
        _parameter_shapes_match(restored_coefs, best_coefs)
        and _parameter_shapes_match(restored_intercepts, best_intercepts)
        and all(
            np.array_equal(np.asarray(restored, dtype=np.float64), np.asarray(source, dtype=np.float64))
            for restored, source in zip(restored_coefs, best_coefs)
        )
        and all(
            np.array_equal(np.asarray(restored, dtype=np.float64), np.asarray(source, dtype=np.float64))
            for restored, source in zip(restored_intercepts, best_intercepts)
        )
    )


def _best_state_valid(
    result: object,
    last_valid_score: float,
    best_validation_score: float,
    best_coefs: Sequence[WeightArray],
    best_intercepts: Sequence[WeightArray],
    coefs: Sequence[WeightArray],
    intercepts: Sequence[WeightArray],
) -> bool:
    if not (isinstance(result, tuple) and len(result) == 3):
        return False
    next_score, next_best_coefs, next_best_intercepts = result
    if not (
        _finite_or_neg_inf_scalar(next_score)
        and _parameter_tuple_valid(next_best_coefs)
        and _parameter_tuple_valid(next_best_intercepts)
    ):
        return False
    improved = float(last_valid_score) > float(best_validation_score)
    expected_score = float(last_valid_score) if improved else float(best_validation_score)
    expected_coefs = coefs if improved else best_coefs
    expected_intercepts = intercepts if improved else best_intercepts
    return bool(
        np.isclose(float(next_score), expected_score)
        and _parameter_shapes_match(next_best_coefs, expected_coefs)
        and _parameter_shapes_match(next_best_intercepts, expected_intercepts)
        and all(
            np.array_equal(np.asarray(next_value, dtype=np.float64), np.asarray(expected, dtype=np.float64))
            for next_value, expected in zip(next_best_coefs, expected_coefs)
        )
        and all(
            np.array_equal(np.asarray(next_value, dtype=np.float64), np.asarray(expected, dtype=np.float64))
            for next_value, expected in zip(next_best_intercepts, expected_intercepts)
        )
    )


@register_atom(witness_mlp_stochastic_validation_targets)
@icontract.require(lambda y_val: _target_matrix_valid(y_val), "y_val must be a finite nonempty numeric or boolean target array")
@icontract.require(lambda is_classifier: _bool_valid(is_classifier), "is_classifier must be boolean")
@icontract.require(
    lambda label_binarizer_state=None, is_classifier=False: (not is_classifier)
    or (label_binarizer_state is not None and _label_binarizer_state_valid(label_binarizer_state)),
    "classifier validation-target decoding requires a fitted label-binarizer state",
)
@icontract.ensure(
    lambda result, y_val: (sp.issparse(result) and result.shape[0] == np.asarray(y_val).shape[0])
    or np.asarray(result).shape[0] == np.asarray(y_val).shape[0],
    "validation targets must preserve sample count",
)
def mlp_stochastic_validation_targets(
    y_val: ValidationTargets,
    *,
    is_classifier: bool,
    label_binarizer_state: LabelBinarizerState | None = None,
) -> DecodedTargets:
    """Resolve sklearn's early-stopping validation targets after optional classifier decoding."""
    values = np.asarray(y_val)
    if not is_classifier:
        return np.asarray(values, dtype=np.float64)
    decoded = label_binarizer_inverse_transform(values, label_binarizer_state)
    if sp.issparse(decoded):
        return decoded.tocsr()
    return np.asarray(decoded)


@register_atom(witness_mlp_validation_scores_append)
@icontract.require(lambda validation_scores: _validation_scores_valid(validation_scores), "validation_scores must be a tuple of finite validation scores")
@icontract.require(lambda val_score: _finite_scalar(val_score), "val_score must be finite")
@icontract.ensure(
    lambda result, validation_scores, val_score: isinstance(result, tuple)
    and result[:-1] == validation_scores
    and len(result) == len(validation_scores) + 1
    and np.isclose(float(result[-1]), float(val_score)),
    "updated validation-score history must append val_score once",
)
def mlp_validation_scores_append(
    validation_scores: tuple[float, ...],
    val_score: float,
) -> tuple[float, ...]:
    """Append one validation score to sklearn's early-stopping score history."""
    return tuple(validation_scores) + (float(val_score),)


@register_atom(witness_mlp_monitor_best_state)
@icontract.require(lambda last_valid_score: _finite_scalar(last_valid_score), "last_valid_score must be finite")
@icontract.require(lambda best_validation_score: _finite_or_neg_inf_scalar(best_validation_score), "best_validation_score must be finite or -inf")
@icontract.require(
    lambda best_coefs, best_intercepts, coefs, intercepts: _parameter_state_inputs_valid(
        best_coefs, best_intercepts, coefs, intercepts
    ),
    "best and current parameter sequences must be finite and shape-aligned",
)
@icontract.ensure(
    lambda result, last_valid_score, best_validation_score, best_coefs, best_intercepts, coefs, intercepts: _best_state_valid(
        result,
        last_valid_score,
        best_validation_score,
        best_coefs,
        best_intercepts,
        coefs,
        intercepts,
    ),
    "best-state update must preserve or replace cached score and parameters according to sklearn's improvement rule",
)
def mlp_monitor_best_state(
    last_valid_score: float,
    best_validation_score: float,
    best_coefs: Sequence[WeightArray],
    best_intercepts: Sequence[WeightArray],
    coefs: Sequence[WeightArray],
    intercepts: Sequence[WeightArray],
) -> tuple[float, ParameterTuple, ParameterTuple]:
    """Update sklearn's cached best validation score and parameter copies."""
    if float(last_valid_score) > float(best_validation_score):
        return (
            float(last_valid_score),
            tuple(np.array(coef, dtype=np.float64, copy=True) for coef in coefs),
            tuple(np.array(intercept, dtype=np.float64, copy=True) for intercept in intercepts),
        )
    return (
        float(best_validation_score),
        tuple(np.asarray(coef, dtype=np.float64) for coef in best_coefs),
        tuple(np.asarray(intercept, dtype=np.float64) for intercept in best_intercepts),
    )


@register_atom(witness_mlp_restore_best_parameters)
@icontract.require(
    lambda best_coefs, best_intercepts: _parameter_sequence_valid(best_coefs)
    and _parameter_sequence_valid(best_intercepts)
    and len(best_coefs) == len(best_intercepts),
    "best_coefs and best_intercepts must be finite parameter sequences with matching layer counts",
)
@icontract.ensure(
    lambda result, best_coefs, best_intercepts: _restored_parameters_valid(result, best_coefs, best_intercepts),
    "restored parameter tuples must match the cached best parameters",
)
def mlp_restore_best_parameters(
    best_coefs: Sequence[WeightArray],
    best_intercepts: Sequence[WeightArray],
) -> tuple[ParameterTuple, ParameterTuple]:
    """Materialize sklearn's final early-stopping parameter restore from cached best state."""
    return (
        tuple(np.asarray(coef, dtype=np.float64) for coef in best_coefs),
        tuple(np.asarray(intercept, dtype=np.float64) for intercept in best_intercepts),
    )
