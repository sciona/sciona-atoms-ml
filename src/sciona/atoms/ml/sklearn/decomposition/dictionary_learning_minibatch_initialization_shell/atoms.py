"""MiniBatchDictionaryLearning initialization-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_dictionary_learning_minibatch_dictionary_buffer,
    witness_dictionary_learning_minibatch_initial_dictionary,
    witness_dictionary_learning_minibatch_resize_dictionary,
)

Matrix = NDArray[np.floating]

def _finite_matrix(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.issubdtype(array.dtype, np.number)
        and np.all(np.isfinite(array))
    )

def _optional_matrix(value: object) -> bool:
    return value is None or _finite_matrix(value)

def _exactly_one_matrix(dict_init: object, svd_dictionary: object) -> bool:
    return (dict_init is None) ^ (svd_dictionary is None)

def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _dtype_name_valid(value: object) -> bool:
    return value in {"float32", "float64"}

def _selected_dictionary_valid(result: object, dict_init: object, svd_dictionary: object) -> bool:
    if not _finite_matrix(result):
        return False
    if dict_init is not None:
        return np.array_equal(np.asarray(result), np.asarray(dict_init))
    return np.array_equal(np.asarray(result), np.asarray(svd_dictionary))

def _resized_dictionary_valid(result: object, dictionary: object, n_components: int) -> bool:
    if not (_finite_matrix(result) and _finite_matrix(dictionary)):
        return False
    result_values = np.asarray(result)
    dictionary_values = np.asarray(dictionary)
    if result_values.shape != (int(n_components), dictionary_values.shape[1]):
        return False
    rank = dictionary_values.shape[0]
    if int(n_components) <= rank:
        return np.array_equal(result_values, dictionary_values[: int(n_components), :])
    pad = int(n_components) - rank
    return bool(
        np.array_equal(result_values[:rank], dictionary_values)
        and np.all(result_values[rank:] == 0)
        and result_values.dtype == dictionary_values.dtype
    )

def _buffer_valid(result: object, dictionary: object, dtype_name: str) -> bool:
    if not (_finite_matrix(result) and _finite_matrix(dictionary) and _dtype_name_valid(dtype_name)):
        return False
    result_values = np.asarray(result)
    dictionary_values = np.asarray(dictionary, dtype=np.dtype(dtype_name))
    return bool(
        result_values.shape == dictionary_values.shape
        and result_values.dtype == np.dtype(dtype_name)
        and result_values.flags["F_CONTIGUOUS"]
        and result_values.flags["WRITEABLE"]
        and np.array_equal(result_values, dictionary_values)
    )

@register_atom(witness_dictionary_learning_minibatch_initial_dictionary)
@icontract.require(lambda dict_init: _optional_matrix(dict_init), "dict_init must be None or a finite numeric matrix")
@icontract.require(lambda svd_dictionary: _optional_matrix(svd_dictionary), "svd_dictionary must be None or a finite numeric matrix")
@icontract.require(lambda dict_init, svd_dictionary: _exactly_one_matrix(dict_init, svd_dictionary), "exactly one of dict_init or svd_dictionary must be provided")
@icontract.ensure(
    lambda result, dict_init, svd_dictionary: _selected_dictionary_valid(result, dict_init, svd_dictionary),
    "result must select the provided dict_init or deferred SVD dictionary input",
)
def dictionary_learning_minibatch_initial_dictionary(
    dict_init: Matrix | None,
    svd_dictionary: Matrix | None,
) -> Matrix:
    """Select sklearn's initial dictionary source before resize and buffer normalization."""
    if dict_init is not None:
        return np.asarray(dict_init)
    assert svd_dictionary is not None
    return np.asarray(svd_dictionary)

@register_atom(witness_dictionary_learning_minibatch_resize_dictionary)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite numeric matrix")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.ensure(
    lambda result, dictionary, n_components: _resized_dictionary_valid(result, dictionary, n_components),
    "result must crop or zero-pad dictionary rows to n_components",
)
def dictionary_learning_minibatch_resize_dictionary(
    dictionary: Matrix,
    n_components: int,
) -> Matrix:
    """Crop or zero-pad an initial dictionary to sklearn's target component count."""
    values = np.asarray(dictionary)
    rank = values.shape[0]
    if int(n_components) <= rank:
        return np.asarray(values[: int(n_components), :])
    return np.concatenate(
        (
            values,
            np.zeros(
                (int(n_components) - rank, values.shape[1]),
                dtype=values.dtype,
            ),
        )
    )

@register_atom(witness_dictionary_learning_minibatch_dictionary_buffer)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite numeric matrix")
@icontract.require(lambda dtype_name: _dtype_name_valid(dtype_name), "dtype_name must be 'float32' or 'float64'")
@icontract.ensure(
    lambda result, dictionary, dtype_name: _buffer_valid(result, dictionary, dtype_name),
    "result must match sklearn's writable Fortran-order dictionary buffer",
)
def dictionary_learning_minibatch_dictionary_buffer(
    dictionary: Matrix,
    dtype_name: str,
) -> Matrix:
    from sklearn.utils.validation import check_array
    """Normalize dictionary storage like sklearn's check_array(..., order='F') plus np.require(..., 'W')."""
    checked = check_array(
        dictionary,
        order="F",
        dtype=np.dtype(dtype_name),
        copy=False,
    )
    return np.require(checked, requirements="W")
