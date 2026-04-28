"""One-vs-rest target encoding helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.preprocessing import LabelBinarizer

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_one_vs_rest_fit_classes,
    witness_one_vs_rest_fit_target_indicator_csc,
    witness_one_vs_rest_partial_fit_target_indicator_csc,
    witness_one_vs_rest_partial_fit_unknown_classes,
    witness_one_vs_rest_target_columns_dense,
)


def _nonempty_targets(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values)
    return bool(array.size >= 1 and array.ndim in {1, 2})


def _class_vector_valid(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.all(np.isfinite(array))
        and np.unique(array).shape[0] == array.shape[0]
    )


def _indicator_matrix_valid(result: sp.csc_matrix, y: NDArray[np.float64]) -> bool:
    targets = np.asarray(y)
    rows = int(targets.shape[0]) if targets.ndim >= 1 else 0
    return bool(sp.isspmatrix_csc(result) and result.shape[0] == rows and result.shape[1] >= 1)


def _classes_match_indicator(result: NDArray[np.float64], indicator: sp.csc_matrix) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (indicator.shape[1], indicator.shape[0]) and np.all(np.isfinite(values)))


def _unknown_classes_valid(result: NDArray[np.float64], classes: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return bool(
        values.ndim == 1
        and np.all(np.isfinite(values))
        and np.unique(values).shape[0] == values.shape[0]
        and np.intersect1d(values, class_values).size == 0
    )


def _target_columns_valid(result: NDArray[np.float64], indicator: sp.csc_matrix) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (indicator.shape[1], indicator.shape[0]) and np.all(np.isin(values, np.array([0.0, 1.0]))))


@register_atom(witness_one_vs_rest_fit_classes)
@icontract.require(lambda y: _nonempty_targets(y), "y must be a nonempty 1D or 2D target array")
@icontract.ensure(lambda result: _class_vector_valid(result), "classes must be a finite unique class vector")
def one_vs_rest_fit_classes(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fit sklearn's sparse LabelBinarizer and return the discovered class vector."""
    label_binarizer = LabelBinarizer(sparse_output=True)
    label_binarizer.fit(np.asarray(y))
    return np.asarray(label_binarizer.classes_, dtype=np.float64)


@register_atom(witness_one_vs_rest_fit_target_indicator_csc)
@icontract.require(lambda y: _nonempty_targets(y), "y must be a nonempty 1D or 2D target array")
@icontract.ensure(lambda result, y: _indicator_matrix_valid(result, y), "indicator must be a CSC target matrix with one row per sample")
def one_vs_rest_fit_target_indicator_csc(y: NDArray[np.float64]) -> sp.csc_matrix:
    """Fit sklearn's sparse LabelBinarizer and return the CSC target indicator matrix."""
    label_binarizer = LabelBinarizer(sparse_output=True)
    return label_binarizer.fit_transform(np.asarray(y)).tocsc()


@register_atom(witness_one_vs_rest_partial_fit_unknown_classes)
@icontract.require(lambda y: _nonempty_targets(y), "y must be a nonempty 1D or 2D target array")
@icontract.require(lambda classes: _class_vector_valid(classes), "classes must be a finite unique class vector")
@icontract.ensure(lambda result, classes: _unknown_classes_valid(result, classes), "unknown classes must be finite, unique, and disjoint from classes")
def one_vs_rest_partial_fit_unknown_classes(
    y: NDArray[np.float64],
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return sorted unique labels present in y but absent from the known class vector."""
    targets = np.asarray(y, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return np.asarray(np.setdiff1d(np.unique(targets), class_values), dtype=np.float64)


@register_atom(witness_one_vs_rest_partial_fit_target_indicator_csc)
@icontract.require(lambda y: _nonempty_targets(y), "y must be a nonempty 1D or 2D target array")
@icontract.require(lambda classes: _class_vector_valid(classes), "classes must be a finite unique class vector")
@icontract.require(lambda y, classes: np.asarray(one_vs_rest_partial_fit_unknown_classes(y, classes)).size == 0, "y must not contain classes outside the known class vector")
@icontract.ensure(lambda result, y: _indicator_matrix_valid(result, y), "indicator must be a CSC target matrix with one row per sample")
def one_vs_rest_partial_fit_target_indicator_csc(
    y: NDArray[np.float64],
    classes: NDArray[np.float64],
) -> sp.csc_matrix:
    """Fit sklearn's sparse LabelBinarizer on the known classes and transform partial-fit targets to CSC."""
    label_binarizer = LabelBinarizer(sparse_output=True)
    label_binarizer.fit(np.asarray(classes, dtype=np.float64))
    return label_binarizer.transform(np.asarray(y)).tocsc()


@register_atom(witness_one_vs_rest_target_columns_dense)
@icontract.require(lambda indicator: sp.isspmatrix_csc(indicator) and indicator.shape[0] >= 1 and indicator.shape[1] >= 1, "indicator must be a nonempty CSC target matrix")
@icontract.ensure(lambda result, indicator: _target_columns_valid(result, indicator), "target columns must be an output-by-sample 0/1 matrix")
def one_vs_rest_target_columns_dense(
    indicator: sp.csc_matrix,
) -> NDArray[np.float64]:
    """Materialize sklearn's per-column target arrays as an output-by-sample dense matrix."""
    return np.asarray([col.toarray().ravel() for col in indicator.T], dtype=np.float64)
