"""Dictionary-learning wrapper bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dict_learning_online_return_values,
    witness_dict_learning_return_values,
    witness_dictionary_learning_lasso_method,
    witness_dictionary_learning_resolved_n_components,
)

Matrix = NDArray[np.float64]


def _method_valid(value: object) -> bool:
    return value in {"lars", "cd"}


def _optional_positive_int(value: object) -> bool:
    return value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value >= 1
    )


def _matrix_valid(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and np.all(np.isfinite(array)))


def _vector_valid(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and np.all(np.isfinite(array)))


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _resolved_component_count_valid(result: object, X: object, n_components: object) -> bool:
    if not (_matrix_valid(X) and _optional_positive_int(n_components) and _positive_int(result)):
        return False
    X_values = np.asarray(X, dtype=np.float64)
    expected = X_values.shape[1] if n_components is None else int(n_components)
    return int(result) == expected


def _dict_learning_inputs_valid(
    code: object,
    components: object,
    errors: object,
    n_iter: object,
    return_n_iter: object,
) -> bool:
    if not (
        _matrix_valid(code)
        and _matrix_valid(components)
        and _vector_valid(errors)
        and _positive_int(n_iter)
        and isinstance(return_n_iter, bool)
    ):
        return False
    code_values = np.asarray(code, dtype=np.float64)
    components_values = np.asarray(components, dtype=np.float64)
    return code_values.shape[1] == components_values.shape[0]


def _dict_learning_result_valid(
    result: object,
    code: object,
    components: object,
    errors: object,
    n_iter: int,
    return_n_iter: bool,
) -> bool:
    if not _dict_learning_inputs_valid(code, components, errors, n_iter, return_n_iter):
        return False
    code_values = np.asarray(code, dtype=np.float64)
    components_values = np.asarray(components, dtype=np.float64)
    errors_values = np.asarray(errors, dtype=np.float64)
    if not isinstance(result, tuple):
        return False
    if return_n_iter:
        return bool(
            len(result) == 4
            and np.array_equal(np.asarray(result[0], dtype=np.float64), code_values)
            and np.array_equal(np.asarray(result[1], dtype=np.float64), components_values)
            and np.array_equal(np.asarray(result[2], dtype=np.float64), errors_values)
            and int(result[3]) == int(n_iter)
        )
    return bool(
        len(result) == 3
        and np.array_equal(np.asarray(result[0], dtype=np.float64), code_values)
        and np.array_equal(np.asarray(result[1], dtype=np.float64), components_values)
        and np.array_equal(np.asarray(result[2], dtype=np.float64), errors_values)
    )


def _dict_learning_online_inputs_valid(
    components: object,
    return_code: object,
    code: object | None,
) -> bool:
    if not (_matrix_valid(components) and isinstance(return_code, bool)):
        return False
    components_values = np.asarray(components, dtype=np.float64)
    if return_code:
        if not _matrix_valid(code):
            return False
        code_values = np.asarray(code, dtype=np.float64)
        return code_values.shape[1] == components_values.shape[0]
    return code is None or _matrix_valid(code)


def _dict_learning_online_result_valid(
    result: object,
    components: object,
    return_code: bool,
    code: object | None,
) -> bool:
    if not _dict_learning_online_inputs_valid(components, return_code, code):
        return False
    components_values = np.asarray(components, dtype=np.float64)
    if return_code:
        code_values = np.asarray(code, dtype=np.float64)
        return bool(
            isinstance(result, tuple)
            and len(result) == 2
            and np.array_equal(np.asarray(result[0], dtype=np.float64), code_values)
            and np.array_equal(np.asarray(result[1], dtype=np.float64), components_values)
        )
    return np.array_equal(np.asarray(result, dtype=np.float64), components_values)


@register_atom(witness_dictionary_learning_lasso_method)
@icontract.require(lambda method: isinstance(method, str) and _method_valid(method), "method must be 'lars' or 'cd'")
@icontract.ensure(lambda result: result in {"lasso_lars", "lasso_cd"}, "result must be the lasso-prefixed method label")
def dictionary_learning_lasso_method(method: str) -> str:
    """Build sklearn's lasso-prefixed internal method label."""
    return "lasso_" + method


@register_atom(witness_dictionary_learning_resolved_n_components)
@icontract.require(lambda X: _matrix_valid(X), "X must be a finite rank-2 matrix")
@icontract.require(lambda n_components: _optional_positive_int(n_components), "n_components must be None or a positive integer")
@icontract.ensure(
    lambda result, X, n_components: _resolved_component_count_valid(result, X, n_components),
    "result must match sklearn's n_components defaulting rule",
)
def dictionary_learning_resolved_n_components(
    X: Matrix,
    n_components: int | None,
) -> int:
    """Resolve the effective dictionary component count from X and n_components."""
    if n_components is None:
        return int(np.asarray(X).shape[1])
    return int(n_components)


@register_atom(witness_dict_learning_return_values)
@icontract.require(
    lambda code, components, errors, n_iter, return_n_iter: _dict_learning_inputs_valid(code, components, errors, n_iter, return_n_iter),
    "dict_learning return inputs must be shape-compatible and finite",
)
@icontract.ensure(
    lambda result, code, components, errors, n_iter, return_n_iter: _dict_learning_result_valid(
        result, code, components, errors, n_iter, return_n_iter
    ),
    "result must match sklearn's return_n_iter packaging",
)
def dict_learning_return_values(
    code: Matrix,
    components: Matrix,
    errors: NDArray[np.float64],
    n_iter: int,
    return_n_iter: bool,
) -> tuple[object, ...]:
    """Package dict_learning's public return tuple."""
    if return_n_iter:
        return code, components, errors, int(n_iter)
    return code, components, errors


@register_atom(witness_dict_learning_online_return_values)
@icontract.require(
    lambda components, return_code, code=None: _dict_learning_online_inputs_valid(components, return_code, code),
    "dict_learning_online return inputs must be finite and shape-compatible",
)
@icontract.ensure(
    lambda result, components, return_code, code=None: _dict_learning_online_result_valid(
        result, components, return_code, code
    ),
    "result must match sklearn's return_code packaging",
)
def dict_learning_online_return_values(
    components: Matrix,
    return_code: bool,
    code: Matrix | None = None,
) -> Matrix | tuple[Matrix, Matrix]:
    """Package dict_learning_online's public return value."""
    if return_code:
        assert code is not None
        return code, components
    return components
