"""FeatureHasher shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_feature_hasher_csr_matrix,
    witness_feature_hasher_dict_items,
    witness_feature_hasher_pair_items,
    witness_feature_hasher_require_nonempty_samples,
    witness_feature_hasher_sample_count,
    witness_feature_hasher_string_items,
)

PairItem = tuple[str, float]
PairSample = tuple[PairItem, ...]
PairSamples = tuple[PairSample, ...]


def _finite_numeric(value: object) -> bool:
    return bool(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _dict_samples_valid(raw_samples: tuple[dict[str, object], ...]) -> bool:
    if not isinstance(raw_samples, tuple) or len(raw_samples) < 1:
        return False
    for sample in raw_samples:
        if not isinstance(sample, dict):
            return False
        for key, value in sample.items():
            if not isinstance(key, str) or not _finite_numeric(value):
                return False
    return True


def _pair_samples_valid(raw_samples: PairSamples) -> bool:
    if not isinstance(raw_samples, tuple) or len(raw_samples) < 1:
        return False
    for sample in raw_samples:
        if not isinstance(sample, tuple):
            return False
        for pair in sample:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not _finite_numeric(pair[1])
            ):
                return False
    return True


def _string_samples_valid(raw_samples: tuple[tuple[str, ...] | str, ...]) -> bool:
    if not isinstance(raw_samples, tuple) or len(raw_samples) < 1:
        return False
    for sample in raw_samples:
        if isinstance(sample, str):
            return False
        if not isinstance(sample, tuple):
            return False
        if not all(isinstance(token, str) for token in sample):
            return False
    return True


def _indptr_valid(indptr: NDArray[np.int64]) -> bool:
    values = np.asarray(indptr)
    if values.ndim != 1 or values.shape[0] < 1 or not np.issubdtype(values.dtype, np.integer):
        return False
    if int(values[0]) != 0 or np.any(values < 0):
        return False
    return bool(np.all(values[1:] >= values[:-1]))


def _dtype_name_valid(dtype_name: str) -> bool:
    if not isinstance(dtype_name, str) or not dtype_name:
        return False
    try:
        dtype = np.dtype(dtype_name)
    except TypeError:
        return False
    return bool(dtype.kind not in {"b", "u"})


def _csr_inputs_valid(
    indices: NDArray[np.int64],
    indptr: NDArray[np.int64],
    values: NDArray[np.float64],
    n_features: int,
    dtype_name: str,
) -> bool:
    indices_values = np.asarray(indices)
    indptr_values = np.asarray(indptr)
    data_values = np.asarray(values, dtype=np.float64)
    if not _indptr_valid(indptr):
        return False
    if indices_values.ndim != 1 or data_values.ndim != 1:
        return False
    if not np.issubdtype(indices_values.dtype, np.integer):
        return False
    if indices_values.shape[0] != data_values.shape[0]:
        return False
    if int(indptr_values[-1]) != indices_values.shape[0]:
        return False
    if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
        return False
    if np.any(indices_values < 0) or np.any(indices_values >= n_features):
        return False
    if not np.all(np.isfinite(data_values)):
        return False
    return _dtype_name_valid(dtype_name)


def _pair_samples_result_valid(result: PairSamples) -> bool:
    return _pair_samples_valid(result)


def _csr_result_valid(
    result: csr_matrix,
    indptr: NDArray[np.int64],
    n_features: int,
    dtype_name: str,
) -> bool:
    dtype = np.dtype(dtype_name)
    expected_rows = int(np.asarray(indptr, dtype=np.int64).shape[0] - 1)
    return bool(
        isinstance(result, csr_matrix)
        and result.shape == (expected_rows, int(n_features))
        and result.dtype == dtype
    )


@register_atom(witness_feature_hasher_dict_items)
@icontract.require(lambda raw_samples: _dict_samples_valid(raw_samples), "raw_samples must be a nonempty tuple of dictionaries with string keys and finite numeric values")
@icontract.ensure(lambda result: _pair_samples_result_valid(result), "dict-mode normalization must return finite float-valued pair samples")
def feature_hasher_dict_items(
    raw_samples: tuple[dict[str, object], ...],
) -> PairSamples:
    """Convert dict input samples to per-sample key-value tuples."""
    return tuple(
        tuple((str(key), float(value)) for key, value in sample.items())
        for sample in raw_samples
    )


@register_atom(witness_feature_hasher_pair_items)
@icontract.require(lambda raw_samples: _pair_samples_valid(raw_samples), "raw_samples must be a nonempty tuple of per-sample string/numeric pairs")
@icontract.ensure(lambda result: _pair_samples_result_valid(result), "pair-mode normalization must return finite float-valued pair samples")
def feature_hasher_pair_items(
    raw_samples: PairSamples,
) -> PairSamples:
    """Normalize pair-mode input samples to float-valued per-sample tuples."""
    return tuple(
        tuple((feature_name, float(value)) for feature_name, value in sample)
        for sample in raw_samples
    )


@register_atom(witness_feature_hasher_string_items)
@icontract.require(lambda raw_samples: _string_samples_valid(raw_samples), "raw_samples must be a nonempty tuple of non-string iterables of strings")
@icontract.ensure(lambda result: _pair_samples_result_valid(result), "string-mode normalization must return unit-weight pair samples")
def feature_hasher_string_items(
    raw_samples: tuple[tuple[str, ...] | str, ...],
) -> PairSamples:
    """Convert string-mode samples to per-token unit-weight key-value tuples."""
    return tuple(
        tuple((token, 1.0) for token in sample)
        for sample in raw_samples
    )


@register_atom(witness_feature_hasher_sample_count)
@icontract.require(lambda indptr: _indptr_valid(indptr), "indptr must be a nonempty 1D nondecreasing integer vector starting at zero")
@icontract.ensure(lambda result: isinstance(result, int) and result >= 0, "sample count must be nonnegative")
def feature_hasher_sample_count(
    indptr: NDArray[np.int64],
) -> int:
    """Compute the number of samples encoded by CSR indptr."""
    values = np.asarray(indptr, dtype=np.int64)
    return int(values.shape[0] - 1)


@register_atom(witness_feature_hasher_require_nonempty_samples)
@icontract.require(lambda n_samples: isinstance(n_samples, int) and not isinstance(n_samples, bool) and n_samples >= 0, "n_samples must be a nonnegative integer")
@icontract.ensure(lambda result: result >= 1, "nonempty sample count must be positive")
def feature_hasher_require_nonempty_samples(
    n_samples: int,
) -> int:
    """Raise if the hashed sample stream is empty."""
    if n_samples == 0:
        raise ValueError("Cannot vectorize empty sequence.")
    return int(n_samples)


@register_atom(witness_feature_hasher_csr_matrix)
@icontract.require(
    lambda indices, indptr, values, n_features, dtype_name="float64": _csr_inputs_valid(
        indices, indptr, values, n_features, dtype_name
    ),
    "indices, indptr, values, n_features, and dtype_name must define a valid finite CSR payload",
)
@icontract.ensure(
    lambda result, indptr, n_features, dtype_name="float64": _csr_result_valid(
        result, indptr, n_features, dtype_name
    ),
    "CSR assembly must preserve the expected output shape and dtype",
)
def feature_hasher_csr_matrix(
    indices: NDArray[np.int64],
    indptr: NDArray[np.int64],
    values: NDArray[np.float64],
    *,
    n_features: int,
    dtype_name: str = "float64",
) -> csr_matrix:
    """Build and duplicate-sum the CSR output matrix from hashed payload arrays."""
    dtype = np.dtype(dtype_name)
    matrix = csr_matrix(
        (
            np.asarray(values, dtype=dtype),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        dtype=dtype,
        shape=(feature_hasher_require_nonempty_samples(feature_hasher_sample_count(indptr)), int(n_features)),
    )
    matrix.sum_duplicates()
    return matrix
