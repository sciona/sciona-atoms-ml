"""Selected semi-supervised atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import laplacian as csgraph_laplacian
from sklearn.utils.multiclass import check_classification_targets

from sciona.ghost.registry import register_atom

from .state_models import GraphKernel, LabelPropagationState, MatrixLike
from .witnesses import (
    witness_label_propagation_fit,
    witness_label_propagation_predict,
    witness_label_propagation_predict_proba,
    witness_label_spreading_fit,
    witness_label_spreading_predict,
    witness_label_spreading_predict_proba,
)

LabelVector = NDArray[np.object_] | NDArray[np.int_] | list[object] | tuple[object, ...]


def _is_2d(X: MatrixLike) -> bool:
    return bool(getattr(X, "ndim", 0) == 2)


def _sample_count(X: MatrixLike) -> int:
    return int(X.shape[0])


def _feature_count(X: MatrixLike) -> int:
    return int(X.shape[1])


def _is_label_vector(y: LabelVector) -> bool:
    return bool(np.asarray(y).ndim == 1)


def _same_sample_count(X: MatrixLike, y: LabelVector) -> bool:
    return _sample_count(X) == int(np.asarray(y).shape[0])


def _kernel_valid(kernel: GraphKernel) -> bool:
    return kernel in {"rbf", "knn"} if isinstance(kernel, str) else callable(kernel)


def _n_neighbors_valid(n_neighbors: int) -> bool:
    return isinstance(n_neighbors, int) and n_neighbors > 0


def _positive_int(value: int) -> bool:
    return isinstance(value, int) and value > 0


def _nonnegative_float(value: float) -> bool:
    return float(value) >= 0.0


def _alpha_valid(alpha: float) -> bool:
    return 0.0 < float(alpha) < 1.0


def _state_valid(state: LabelPropagationState) -> bool:
    return bool(
        _is_2d(state.X)
        and state.classes.ndim == 1
        and state.classes.shape[0] > 0
        and state.label_distributions.shape == (_sample_count(state.X), state.classes.shape[0])
        and state.transduction.shape == (_sample_count(state.X),)
        and state.n_iter >= 0
        and _kernel_valid(state.kernel)
        and _n_neighbors_valid(state.n_neighbors)
        and state.variant in {"propagation", "spreading"}
        and state.n_features_in == _feature_count(state.X)
        and np.all(np.isfinite(state.label_distributions))
    )


def _proba_valid(result: NDArray[np.float64], X: MatrixLike, state: LabelPropagationState) -> bool:
    row_sums = np.sum(result, axis=1)
    return bool(
        result.shape == (_sample_count(X), state.classes.shape[0])
        and np.all(np.isfinite(result))
        and np.all(result >= 0.0)
        and np.allclose(row_sums, np.ones_like(row_sums))
    )


def _prediction_valid(result: NDArray[np.object_], X: MatrixLike, state: LabelPropagationState) -> bool:
    return bool(result.shape == (_sample_count(X),) and np.isin(result, state.classes).all())


def _get_kernel(
    X_fit: MatrixLike,
    X_query: MatrixLike | None,
    *,
    kernel: GraphKernel,
    gamma: float,
    n_neighbors: int,
    n_jobs: int | None,
) -> MatrixLike | NDArray[np.int_]:
    if kernel == "rbf":
        return rbf_kernel(X_fit, X_fit if X_query is None else X_query, gamma=gamma)
    if kernel == "knn":
        nn_fit = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs).fit(X_fit)
        if X_query is None:
            return nn_fit.kneighbors_graph(nn_fit._fit_X, n_neighbors, mode="connectivity")
        return nn_fit.kneighbors(X_query, return_distance=False)
    assert callable(kernel)
    return kernel(X_fit, X_fit if X_query is None else X_query)


def _build_graph(
    X: MatrixLike,
    *,
    variant: str,
    kernel: GraphKernel,
    gamma: float,
    n_neighbors: int,
    n_jobs: int | None,
) -> MatrixLike:
    affinity_matrix = _get_kernel(
        X,
        None,
        kernel=kernel,
        gamma=gamma,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    if variant == "propagation":
        normalizer = affinity_matrix.sum(axis=1)
        if sparse.isspmatrix(affinity_matrix):
            normalizer = np.ravel(normalizer)
            return sparse.diags(1.0 / normalizer) @ affinity_matrix
        return np.asarray(affinity_matrix, dtype=np.float64) / normalizer[:, np.newaxis]

    laplacian = csgraph_laplacian(affinity_matrix, normed=True)
    laplacian = -laplacian
    if sparse.issparse(laplacian):
        diag_mask = laplacian.row == laplacian.col
        laplacian.data[diag_mask] = 0.0
    else:
        laplacian.flat[:: X.shape[0] + 1] = 0.0
    return laplacian


def _fit_graph_labels(
    X: MatrixLike,
    y: LabelVector,
    *,
    variant: str,
    kernel: GraphKernel,
    gamma: float,
    n_neighbors: int,
    alpha: float | None,
    max_iter: int,
    tol: float,
    n_jobs: int | None,
) -> LabelPropagationState:
    checked_x = check_array(X, accept_sparse=["csr", "csc"])
    checked_y = np.asarray(y)
    check_consistent_length(checked_x, checked_y)
    check_classification_targets(checked_y)

    graph_matrix = _build_graph(
        checked_x,
        variant=variant,
        kernel=kernel,
        gamma=gamma,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    classes = np.asarray(np.unique(checked_y), dtype=object)
    classes = classes[classes != -1]
    n_samples, n_classes = len(checked_y), len(classes)
    unlabeled = checked_y == -1

    label_distributions = np.zeros((n_samples, n_classes), dtype=np.float64)
    for label in classes:
        label_distributions[checked_y == label, classes == label] = 1.0

    y_static = np.copy(label_distributions)
    if variant == "propagation":
        y_static[unlabeled] = 0.0
    else:
        assert alpha is not None
        y_static *= 1.0 - alpha

    previous = np.zeros((checked_x.shape[0], n_classes), dtype=np.float64)
    unlabeled_mask = unlabeled[:, np.newaxis]
    if sparse.issparse(graph_matrix):
        graph_matrix = graph_matrix.tocsr()

    n_iter = 0
    for n_iter in range(max_iter):
        if np.abs(label_distributions - previous).sum() < tol:
            break
        previous = label_distributions
        label_distributions = safe_sparse_dot(graph_matrix, label_distributions)
        if variant == "propagation":
            normalizer = np.sum(label_distributions, axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            label_distributions /= normalizer
            label_distributions = np.where(unlabeled_mask, label_distributions, y_static)
        else:
            assert alpha is not None
            label_distributions = np.multiply(alpha, label_distributions) + y_static
    else:
        warnings.warn("max_iter=%d was reached without convergence." % max_iter, category=ConvergenceWarning)
        n_iter += 1

    normalizer = np.sum(label_distributions, axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.0] = 1.0
    label_distributions /= normalizer
    transduction = classes[np.argmax(label_distributions, axis=1)].ravel()
    return LabelPropagationState(
        X=checked_x,
        classes=classes,
        label_distributions=np.asarray(label_distributions, dtype=np.float64),
        transduction=np.asarray(transduction, dtype=object),
        n_iter=int(n_iter),
        kernel=kernel,
        gamma=float(gamma),
        n_neighbors=int(n_neighbors),
        alpha=alpha,
        variant=variant,
        n_jobs=n_jobs,
        n_features_in=int(checked_x.shape[1]),
    )


def _predict_proba_from_state(X: MatrixLike, state: LabelPropagationState) -> NDArray[np.float64]:
    checked_x = check_array(X, accept_sparse=["csc", "csr", "coo", "dok", "bsr", "lil", "dia"])
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    weight_matrices = _get_kernel(
        state.X,
        checked_x,
        kernel=state.kernel,
        gamma=state.gamma,
        n_neighbors=state.n_neighbors,
        n_jobs=state.n_jobs,
    )
    if state.kernel == "knn":
        probabilities = np.array([np.sum(state.label_distributions[weight_matrix], axis=0) for weight_matrix in weight_matrices])
    else:
        probabilities = safe_sparse_dot(weight_matrices.T, state.label_distributions)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    normalizer = np.atleast_2d(np.sum(probabilities, axis=1)).T
    probabilities /= normalizer
    return probabilities


@register_atom(witness_label_propagation_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D feature matrix")
@icontract.require(lambda y: _is_label_vector(y), "y must be a 1D label vector")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have equal sample count")
@icontract.require(lambda kernel: _kernel_valid(kernel), "kernel must be 'rbf', 'knn', or callable")
@icontract.require(lambda n_neighbors: _n_neighbors_valid(n_neighbors), "n_neighbors must be positive")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda tol: _nonnegative_float(tol), "tol must be non-negative")
@icontract.ensure(lambda result: _state_valid(result), "fitted propagation state must contain class distributions")
def label_propagation_fit(
    X: MatrixLike,
    y: LabelVector,
    *,
    kernel: GraphKernel = "rbf",
    gamma: float = 20.0,
    n_neighbors: int = 7,
    max_iter: int = 1000,
    tol: float = 1e-3,
    n_jobs: int | None = None,
) -> LabelPropagationState:
    """Fit hard-clamped graph label propagation and return immutable state."""
    return _fit_graph_labels(
        X,
        y,
        variant="propagation",
        kernel=kernel,
        gamma=gamma,
        n_neighbors=n_neighbors,
        alpha=None,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
    )


@register_atom(witness_label_propagation_predict_proba)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D feature matrix")
@icontract.require(lambda state: _state_valid(state), "state must contain fitted propagation distributions")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _proba_valid(result, X, state), "probability rows must sum to one")
def label_propagation_predict_proba(
    X: MatrixLike,
    state: LabelPropagationState,
) -> NDArray[np.float64]:
    """Predict class probabilities from a fitted label-propagation state."""
    return _predict_proba_from_state(X, state)


@register_atom(witness_label_propagation_predict)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D feature matrix")
@icontract.require(lambda state: _state_valid(state), "state must contain fitted propagation distributions")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _prediction_valid(result, X, state), "predictions must be fitted classes")
def label_propagation_predict(
    X: MatrixLike,
    state: LabelPropagationState,
) -> NDArray[np.object_]:
    """Predict class labels from a fitted label-propagation state."""
    probabilities = label_propagation_predict_proba(X, state)
    return np.asarray(state.classes[np.argmax(probabilities, axis=1)].ravel(), dtype=object)


@register_atom(witness_label_spreading_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D feature matrix")
@icontract.require(lambda y: _is_label_vector(y), "y must be a 1D label vector")
@icontract.require(lambda X, y: _same_sample_count(X, y), "X and y must have equal sample count")
@icontract.require(lambda kernel: _kernel_valid(kernel), "kernel must be 'rbf', 'knn', or callable")
@icontract.require(lambda n_neighbors: _n_neighbors_valid(n_neighbors), "n_neighbors must be positive")
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must be between zero and one")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda tol: _nonnegative_float(tol), "tol must be non-negative")
@icontract.ensure(lambda result: _state_valid(result), "fitted spreading state must contain class distributions")
def label_spreading_fit(
    X: MatrixLike,
    y: LabelVector,
    *,
    kernel: GraphKernel = "rbf",
    gamma: float = 20.0,
    n_neighbors: int = 7,
    alpha: float = 0.2,
    max_iter: int = 30,
    tol: float = 1e-3,
    n_jobs: int | None = None,
) -> LabelPropagationState:
    """Fit soft-clamped graph label spreading and return immutable state."""
    return _fit_graph_labels(
        X,
        y,
        variant="spreading",
        kernel=kernel,
        gamma=gamma,
        n_neighbors=n_neighbors,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
    )


@register_atom(witness_label_spreading_predict_proba)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D feature matrix")
@icontract.require(lambda state: _state_valid(state), "state must contain fitted spreading distributions")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _proba_valid(result, X, state), "probability rows must sum to one")
def label_spreading_predict_proba(
    X: MatrixLike,
    state: LabelPropagationState,
) -> NDArray[np.float64]:
    """Predict class probabilities from a fitted label-spreading state."""
    return _predict_proba_from_state(X, state)


@register_atom(witness_label_spreading_predict)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D feature matrix")
@icontract.require(lambda state: _state_valid(state), "state must contain fitted spreading distributions")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _prediction_valid(result, X, state), "predictions must be fitted classes")
def label_spreading_predict(
    X: MatrixLike,
    state: LabelPropagationState,
) -> NDArray[np.object_]:
    """Predict class labels from a fitted label-spreading state."""
    probabilities = label_spreading_predict_proba(X, state)
    return np.asarray(state.classes[np.argmax(probabilities, axis=1)].ravel(), dtype=object)
