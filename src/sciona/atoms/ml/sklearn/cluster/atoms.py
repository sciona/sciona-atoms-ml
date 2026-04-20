"""Selected clustering atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import euclidean_distances, pairwise_distances_argmin
from sklearn.utils import check_array, check_random_state

from sciona.ghost.registry import register_atom

from .state_models import AffinityPropagationState
from .witnesses import (
    witness_affinity_propagation,
    witness_affinity_propagation_fit,
    witness_affinity_propagation_predict,
)

MatrixLike = NDArray[np.float64] | sp.spmatrix | list[list[float]]
PreferenceLike = float | NDArray[np.float64] | list[float] | None
RandomStateLike = int | np.random.RandomState | None
AffinityPropagationResult = tuple[NDArray[np.int_], NDArray[np.int_]]
AffinityPropagationResultWithIter = tuple[NDArray[np.int_], NDArray[np.int_], int]
AffinityPropagationOutput = AffinityPropagationResult | AffinityPropagationResultWithIter


def _is_2d_matrix(X: MatrixLike) -> bool:
    if sp.issparse(X):
        return bool(X.ndim == 2)
    return bool(np.asarray(X).ndim == 2)


def _sample_count(X: MatrixLike) -> int:
    return int(X.shape[0]) if sp.issparse(X) else int(np.asarray(X).shape[0])


def _feature_count(X: MatrixLike) -> int:
    return int(X.shape[1]) if sp.issparse(X) else int(np.asarray(X).shape[1])


def _is_square_matrix(X: MatrixLike) -> bool:
    if not _is_2d_matrix(X):
        return False
    shape = X.shape if sp.issparse(X) else np.asarray(X).shape
    return bool(shape[0] == shape[1])


def _damping_valid(damping: float) -> bool:
    return bool(0.5 <= float(damping) < 1.0)


def _positive_int(value: int) -> bool:
    return isinstance(value, int) and value >= 1


def _affinity_valid(affinity: str) -> bool:
    return affinity in {"euclidean", "precomputed"}


def _preference_valid(preference: PreferenceLike) -> bool:
    if preference is None:
        return True
    values = np.asarray(preference, dtype=np.float64)
    return bool(values.ndim <= 1 and np.all(np.isfinite(values)))


def _preference_matches_samples(preference: PreferenceLike, X: MatrixLike) -> bool:
    if preference is None:
        return True
    values = np.asarray(preference, dtype=np.float64)
    return bool(values.ndim == 0 or values.shape == (_sample_count(X),))


def _equal_similarities_and_preferences(S: NDArray[np.float64], preference: NDArray[np.float64]) -> bool:
    return bool(np.all(S == S.flat[0]) and np.all(preference == preference.flat[0]))


def _prepare_preference(preference: PreferenceLike, affinity_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    if preference is None:
        return np.asarray(np.median(affinity_matrix), dtype=np.float64)
    values = np.asarray(preference, dtype=np.float64)
    if values.ndim > 1:
        raise ValueError("preference must be a scalar or a vector")
    if values.ndim == 1 and values.shape[0] != affinity_matrix.shape[0]:
        raise ValueError("preference length must equal the sample count")
    return values


def _stored_preference(preference: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if preference.ndim == 0:
        return float(preference)
    return np.asarray(preference, dtype=np.float64).copy()


def _indices_and_labels_valid(centers: NDArray[np.int_], labels: NDArray[np.int_], n_samples: int) -> bool:
    if centers.ndim != 1 or labels.shape != (n_samples,):
        return False
    if centers.size > 0 and (centers.min() < 0 or centers.max() >= n_samples):
        return False
    return bool(np.all((labels == -1) | ((labels >= 0) & (labels < max(centers.size, 1)))))


def _affinity_result_valid(result: AffinityPropagationOutput, S: MatrixLike, return_n_iter: bool) -> bool:
    expected_len = 3 if return_n_iter else 2
    if len(result) != expected_len:
        return False
    centers = np.asarray(result[0])
    labels = np.asarray(result[1])
    if centers.dtype.kind not in {"i", "u"} or labels.dtype.kind not in {"i", "u"}:
        return False
    if return_n_iter and (not isinstance(result[2], int) or result[2] < 0):
        return False
    return _indices_and_labels_valid(centers.astype(np.int_), labels.astype(np.int_), _sample_count(S))


def _state_valid(state: AffinityPropagationState) -> bool:
    n_samples = state.affinity_matrix.shape[0]
    centers_ok = _indices_and_labels_valid(state.cluster_centers_indices, state.labels, n_samples)
    centers_shape_ok = (
        state.cluster_centers is None
        if state.affinity == "precomputed"
        else state.cluster_centers is not None
        and state.cluster_centers.shape == (state.cluster_centers_indices.shape[0], state.n_features_in)
    )
    return bool(
        state.affinity_matrix.ndim == 2
        and state.affinity_matrix.shape[0] == state.affinity_matrix.shape[1]
        and state.n_iter >= 0
        and _affinity_valid(state.affinity)
        and _damping_valid(state.damping)
        and state.n_features_in >= 1
        and centers_ok
        and centers_shape_ok
    )


def _prediction_valid(result: NDArray[np.int_], X: MatrixLike) -> bool:
    labels = np.asarray(result)
    return bool(labels.shape == (_sample_count(X),) and labels.dtype.kind in {"i", "u"} and np.all(labels >= -1))


def _as_dense_float_matrix(X: MatrixLike) -> NDArray[np.float64]:
    if sp.issparse(X):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _affinity_propagation_core(
    S: NDArray[np.float64],
    *,
    preference: NDArray[np.float64],
    convergence_iter: int,
    max_iter: int,
    damping: float,
    verbose: bool,
    return_n_iter: bool,
    random_state: np.random.RandomState,
) -> AffinityPropagationOutput:
    n_samples = S.shape[0]
    if n_samples == 1 or _equal_similarities_and_preferences(S, preference):
        warnings.warn(
            "All samples have mutually equal similarities. Returning arbitrary cluster center(s)."
        )
        if preference.flat[0] > S.flat[n_samples - 1]:
            centers = np.arange(n_samples, dtype=np.int_)
            labels = np.arange(n_samples, dtype=np.int_)
        else:
            centers = np.array([0], dtype=np.int_)
            labels = np.array([0] * n_samples, dtype=np.int_)
        if return_n_iter:
            return centers, labels, 0
        return centers, labels

    S.flat[:: n_samples + 1] = preference

    availability = np.zeros((n_samples, n_samples), dtype=np.float64)
    responsibility = np.zeros((n_samples, n_samples), dtype=np.float64)
    tmp = np.zeros((n_samples, n_samples), dtype=np.float64)

    S += (
        np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100
    ) * random_state.standard_normal(size=(n_samples, n_samples))

    exemplars_over_time = np.zeros((n_samples, convergence_iter), dtype=np.float64)
    indices = np.arange(n_samples)
    exemplar_mask = np.zeros(n_samples, dtype=bool)
    never_converged = True
    iteration = 0

    for iteration in range(max_iter):
        np.add(availability, S, tmp)
        best_indices = np.argmax(tmp, axis=1)
        best_values = tmp[indices, best_indices]
        tmp[indices, best_indices] = -np.inf
        second_best_values = np.max(tmp, axis=1)

        np.subtract(S, best_values[:, None], tmp)
        tmp[indices, best_indices] = S[indices, best_indices] - second_best_values

        tmp *= 1.0 - damping
        responsibility *= damping
        responsibility += tmp

        np.maximum(responsibility, 0, out=tmp)
        tmp.flat[:: n_samples + 1] = responsibility.flat[:: n_samples + 1]

        tmp -= np.sum(tmp, axis=0)
        diagonal_availability = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: n_samples + 1] = diagonal_availability

        tmp *= 1.0 - damping
        availability *= damping
        availability -= tmp

        exemplar_mask = (np.diag(availability) + np.diag(responsibility)) > 0
        exemplars_over_time[:, iteration % convergence_iter] = exemplar_mask
        n_exemplars = int(np.sum(exemplar_mask, axis=0))

        if iteration >= convergence_iter:
            stability = np.sum(exemplars_over_time, axis=1)
            unconverged = np.sum((stability == convergence_iter) + (stability == 0)) != n_samples
            if not unconverged and n_exemplars > 0:
                never_converged = False
                if verbose:
                    print(f"Converged after {iteration} iterations.")
                break
    else:
        if verbose:
            print("Did not converge")

    exemplar_indices = np.flatnonzero(exemplar_mask)
    n_exemplars = exemplar_indices.size

    if n_exemplars > 0:
        if never_converged:
            warnings.warn(
                "Affinity propagation did not converge, this model may return degenerate cluster centers and labels.",
                ConvergenceWarning,
            )
        cluster_assignments = np.argmax(S[:, exemplar_indices], axis=1)
        cluster_assignments[exemplar_indices] = np.arange(n_exemplars)
        for cluster_index in range(n_exemplars):
            members = np.asarray(cluster_assignments == cluster_index).nonzero()[0]
            best_member = np.argmax(np.sum(S[members[:, np.newaxis], members], axis=0))
            exemplar_indices[cluster_index] = members[best_member]

        cluster_assignments = np.argmax(S[:, exemplar_indices], axis=1)
        cluster_assignments[exemplar_indices] = np.arange(n_exemplars)
        labels_as_indices = exemplar_indices[cluster_assignments]
        cluster_centers_indices = np.unique(labels_as_indices).astype(np.int_)
        labels = np.searchsorted(cluster_centers_indices, labels_as_indices).astype(np.int_)
    else:
        warnings.warn(
            "Affinity propagation did not converge and this model will not have any cluster centers.",
            ConvergenceWarning,
        )
        labels = np.array([-1] * n_samples, dtype=np.int_)
        cluster_centers_indices = np.array([], dtype=np.int_)

    if return_n_iter:
        return cluster_centers_indices, labels, iteration + 1
    return cluster_centers_indices, labels


@register_atom(witness_affinity_propagation)
@icontract.require(lambda S: _is_square_matrix(S), "S must be a square similarity matrix")
@icontract.require(lambda preference: _preference_valid(preference), "preference must be finite scalar or vector")
@icontract.require(lambda S, preference: _preference_matches_samples(preference, S), "preference length must match samples")
@icontract.require(lambda convergence_iter: _positive_int(convergence_iter), "convergence_iter must be at least one")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be at least one")
@icontract.require(lambda damping: _damping_valid(damping), "damping must be in [0.5, 1.0)")
@icontract.ensure(lambda result, S, return_n_iter: _affinity_result_valid(result, S, return_n_iter), "cluster centers and labels must match sample count")
def affinity_propagation(
    S: MatrixLike,
    *,
    preference: PreferenceLike = None,
    convergence_iter: int = 15,
    max_iter: int = 200,
    damping: float = 0.5,
    copy: bool = True,
    verbose: bool = False,
    return_n_iter: bool = False,
    random_state: RandomStateLike = None,
) -> AffinityPropagationOutput:
    """Run affinity propagation on a square similarity matrix."""
    affinity_matrix = np.asarray(
        check_array(S, dtype=[np.float64, np.float32], copy=copy, force_writeable=True),
        dtype=np.float64,
    )
    if affinity_matrix.shape[0] != affinity_matrix.shape[1]:
        raise ValueError("The matrix of similarities must be a square array.")
    prepared_preference = _prepare_preference(preference, affinity_matrix)
    return _affinity_propagation_core(
        affinity_matrix,
        preference=prepared_preference,
        convergence_iter=convergence_iter,
        max_iter=max_iter,
        damping=damping,
        verbose=verbose,
        return_n_iter=return_n_iter,
        random_state=check_random_state(random_state),
    )


@register_atom(witness_affinity_propagation_fit)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a 2D matrix")
@icontract.require(lambda X, affinity: affinity != "precomputed" or _is_square_matrix(X), "precomputed affinity must be square")
@icontract.require(lambda preference: _preference_valid(preference), "preference must be finite scalar or vector")
@icontract.require(lambda X, preference: _preference_matches_samples(preference, X), "preference length must match samples")
@icontract.require(lambda convergence_iter: _positive_int(convergence_iter), "convergence_iter must be at least one")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be at least one")
@icontract.require(lambda damping: _damping_valid(damping), "damping must be in [0.5, 1.0)")
@icontract.require(lambda affinity: _affinity_valid(affinity), "affinity must be 'euclidean' or 'precomputed'")
@icontract.ensure(lambda result: _state_valid(result), "affinity propagation state must be fitted")
def affinity_propagation_fit(
    X: MatrixLike,
    *,
    damping: float = 0.5,
    max_iter: int = 200,
    convergence_iter: int = 15,
    copy: bool = True,
    preference: PreferenceLike = None,
    affinity: str = "euclidean",
    verbose: bool = False,
    random_state: RandomStateLike = None,
) -> AffinityPropagationState:
    """Fit affinity propagation and return immutable clustering state."""
    if affinity == "precomputed":
        affinity_matrix = np.asarray(
            check_array(X, dtype=[np.float64, np.float32], copy=copy, force_writeable=True),
            dtype=np.float64,
        )
        if affinity_matrix.shape[0] != affinity_matrix.shape[1]:
            raise ValueError("The matrix of similarities must be a square array.")
        n_features_in = int(affinity_matrix.shape[1])
        cluster_centers = None
    else:
        checked_x = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        affinity_matrix = np.asarray(-euclidean_distances(checked_x, squared=True), dtype=np.float64)
        n_features_in = int(checked_x.shape[1])
        cluster_centers = np.empty((0, n_features_in), dtype=np.float64)

    prepared_preference = _prepare_preference(preference, affinity_matrix)
    centers, labels, n_iter = _affinity_propagation_core(
        affinity_matrix,
        max_iter=max_iter,
        convergence_iter=convergence_iter,
        preference=prepared_preference,
        damping=damping,
        verbose=verbose,
        return_n_iter=True,
        random_state=check_random_state(random_state),
    )
    centers = np.asarray(centers, dtype=np.int_)
    labels = np.asarray(labels, dtype=np.int_)

    if affinity != "precomputed":
        selected = checked_x[centers].copy()
        cluster_centers = _as_dense_float_matrix(selected)

    return AffinityPropagationState(
        cluster_centers_indices=centers,
        labels=labels,
        n_iter=int(n_iter),
        affinity_matrix=affinity_matrix,
        cluster_centers=cluster_centers,
        affinity=affinity,
        preference=_stored_preference(prepared_preference),
        damping=float(damping),
        n_features_in=n_features_in,
    )


@register_atom(witness_affinity_propagation_predict)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a 2D matrix")
@icontract.require(lambda state: _state_valid(state), "affinity propagation state must be fitted")
@icontract.require(lambda X, state: state.affinity == "precomputed" or _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _prediction_valid(result, X), "predicted labels must match sample count")
def affinity_propagation_predict(
    X: MatrixLike,
    state: AffinityPropagationState,
) -> NDArray[np.int_]:
    """Assign samples to the nearest fitted affinity-propagation center."""
    if state.affinity == "precomputed":
        raise ValueError("Predict method is not supported when affinity='precomputed'.")
    checked_x = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
    if state.cluster_centers is None:
        raise ValueError("affinity propagation state does not include cluster centers")
    if state.cluster_centers.shape[0] > 0:
        return np.asarray(pairwise_distances_argmin(checked_x, state.cluster_centers), dtype=np.int_)
    warnings.warn(
        "This model does not have any cluster centers because affinity propagation did not converge. Labeling every sample as '-1'.",
        ConvergenceWarning,
    )
    return np.array([-1] * checked_x.shape[0], dtype=np.int_)
