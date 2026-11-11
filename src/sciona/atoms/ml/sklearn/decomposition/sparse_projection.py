"""Reusable sparse randomized projection with explicit numerical controls."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete sparse matrix and numerical controls required')


@register_atom(concrete_only)
def fit_sparse_projection(matrix: object, rank: int, iterations: int,
                          oversamples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit uncentered randomized right singular vectors of finite float64 CSR.

    Rank is a positive Python int less than both matrix dimensions; iterations
    and oversamples are positive Python ints; seed is in [0,2**32). Uses auto
    orientation and power normalization, a randomized range finder and gesdd.
    A dense basis lift uses unoptimized einsum. Returns owned float64 components
    (rank,features) and singular values (rank,), with right-vector sign choice.

    Copies CSR storage while retaining entry order and duplicates. Full input
    is not densified; reduced factors are dense and can approach input size for
    large rank/oversampling. Native pools are bounded to one. Caller owns total
    concurrency, feature ordering, training split and numerical scale. Finite
    input does not guarantee freedom from intermediate overflow; nonfinite
    factors reject. No text, label, vocabulary, centering or scaling policy.
    """
    from sciona._runtime.ml.sparse_randomized_projection import fit_projection
    return fit_projection(matrix, rank, iterations, oversamples, seed)


@register_atom(concrete_only)
def transform_sparse_projection(matrix: object, components: np.ndarray) -> np.ndarray:
    """Project finite float64 CSR rows through finite float64 components.

    Nonempty rows and at least one component are required; component feature
    width matches the sparse matrix. Preserves row order and returns an owned
    dense float64 matrix (rows,components). Copies sparse storage, preserving
    entry order and duplicates; inputs are unchanged. Rejects malformed sparse
    storage and nonfinite input/output. Native pools are limited to one.
    Caller supplies fitted components and feature ordering; no refit, labels,
    split checks, text policy or authentication of component provenance.
    """
    from sciona._runtime.ml.sparse_randomized_projection import transform_projection
    return transform_projection(matrix, components)
