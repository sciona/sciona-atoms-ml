"""Selected SVM helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.extmath import safe_sparse_dot

from sciona.ghost.registry import register_atom

from .witnesses import witness_l1_min_c

MatrixLike = NDArray[np.float64] | sp.spmatrix


def _is_2d(X: MatrixLike) -> bool:
    return bool(getattr(X, "ndim", 0) == 2)


def _is_1d(y: NDArray[np.float64]) -> bool:
    return bool(getattr(y, "ndim", 0) == 1)


def _sample_count(X: MatrixLike) -> int:
    return int(X.shape[0])


def _valid_loss(loss: str) -> bool:
    return loss in {"squared_hinge", "log"}


@register_atom(witness_l1_min_c)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda y: _is_1d(np.asarray(y)), "y must be a 1D vector")
@icontract.require(lambda X, y: _sample_count(X) == np.asarray(y).shape[0], "X and y must have equal sample count")
@icontract.require(lambda loss: _valid_loss(loss), "loss must be 'squared_hinge' or 'log'")
@icontract.require(lambda intercept_scaling: intercept_scaling > 0.0, "intercept_scaling must be positive")
@icontract.ensure(lambda result: result > 0.0, "minimum C bound must be positive")
def l1_min_c(
    X: MatrixLike,
    y: NDArray[np.float64],
    *,
    loss: str = "squared_hinge",
    fit_intercept: bool = True,
    intercept_scaling: float = 1.0,
) -> float:
    """Compute the minimum useful C value for l1-penalized classifiers."""
    checked_x = check_array(X, accept_sparse="csc")
    checked_y = np.asarray(y)
    check_consistent_length(checked_x, checked_y)

    y_matrix = LabelBinarizer(neg_label=-1).fit_transform(checked_y).T
    denominator = np.max(np.abs(safe_sparse_dot(y_matrix, checked_x)))
    if fit_intercept:
        bias = np.full(
            (np.size(checked_y), 1),
            intercept_scaling,
            dtype=np.array(intercept_scaling).dtype,
        )
        denominator = max(denominator, abs(np.dot(y_matrix, bias)).max())

    if denominator == 0.0:
        raise ValueError(
            "Ill-posed l1_min_c calculation: l1 will always select zero coefficients for this data"
        )
    if loss == "squared_hinge":
        return float(0.5 / denominator)
    return float(2.0 / denominator)
