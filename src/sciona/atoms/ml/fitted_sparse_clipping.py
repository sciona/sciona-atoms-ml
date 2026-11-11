"""Training-fitted nonnegative sparse clipping with explicit feature identity.

Registration alone does not qualify catalog publication. The global bound
requires comparable normalized numeric feature coordinates supplied upstream.
"""
import numpy as np
from scipy.sparse import csr_matrix

from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _identity(columns, coordinate, width):
    if (type(columns) is not list or len(columns) != width or width < 1
            or any(type(c) is not str or not c.strip() for c in columns)
            or len(set(columns)) != width):
        raise ValueError('Unique ordered feature identifiers required')
    if type(coordinate) is not str or not coordinate.strip():
        raise ValueError('Explicit common normalized feature coordinate required')


def _matrix(values, columns, coordinate):
    if not isinstance(values, csr_matrix) or values.dtype.kind not in 'biuf':
        raise ValueError('Real numeric CSR matrix required')
    if not values.has_canonical_format:
        raise ValueError('Canonical CSR storage required')
    _identity(columns, coordinate, values.shape[1])


def _abstract(values, columns, coordinate):
    if len(values.shape) != 2 or np.dtype(values.dtype).kind not in 'biuf':
        raise ValueError('Real numeric matrix metadata required')
    _identity(columns, coordinate, values.shape[1])
    if values.dim is not None and not values.dim.is_dimensionless:
        raise ValueError('Normalize feature coordinates before global clipping')


def witness_fit_sparse_clip(training: AbstractArray, feature_columns: list,
        feature_coordinate: str) -> dict:
    _abstract(training, feature_columns, feature_coordinate)
    if training.shape[0] < 1:
        raise ValueError('Nonempty training population required')
    return dict(format='sciona.fitted_sparse_clip.abstract.v1',
                feature_columns=list(feature_columns), feature_coordinate=feature_coordinate)


@register_atom(witness_fit_sparse_clip)
def fit_sparse_clip(training: csr_matrix, feature_columns: list,
        feature_coordinate: str) -> dict:
    """Learn one nonnegative maximum from stored float32 training entries.

    Caller owns feature semantics, comparable normalized coordinates and
    training-only fitting. Implicit zeros are excluded; explicit zeros count.
    Empty/all-NaN stored populations, infinities and negative maxima fail.
    Returns JSON numeric state without retaining training rows.
    """
    from sciona._runtime.ml.fitted_sparse_clip import fit_sparse_clip as fit
    _matrix(training, feature_columns, feature_coordinate)
    state = fit(training)
    return dict(format='sciona.fitted_sparse_clip.v1', feature_columns=list(feature_columns),
                feature_coordinate=feature_coordinate, upper=state.upper)


def witness_apply_sparse_clip(values: AbstractArray, feature_columns: list,
        feature_coordinate: str, state: dict) -> AbstractArray:
    _abstract(values, feature_columns, feature_coordinate)
    if state != dict(format='sciona.fitted_sparse_clip.abstract.v1',
                     feature_columns=feature_columns, feature_coordinate=feature_coordinate):
        raise ValueError('Matching abstract clipping state required')
    return AbstractArray(shape=values.shape, dtype='float32', dim=values.dim)


@register_atom(witness_apply_sparse_clip)
def apply_sparse_clip(values: csr_matrix, feature_columns: list,
        feature_coordinate: str, state: dict) -> csr_matrix:
    """Return independent float32 CSR clipped to the frozen training maximum.

    Exact feature order and coordinate identity must match fitted state. NaNs
    become zero, negative values and negative infinity clip to zero, positive
    infinity clips to the bound. Empty query matrices are supported. No refit,
    densification or executable deserialization occurs.
    """
    from sciona._runtime.ml.fitted_sparse_clip import FittedSparseClip
    _matrix(values, feature_columns, feature_coordinate)
    if (type(state) is not dict or set(state) != {'format', 'feature_columns', 'feature_coordinate', 'upper'}
            or state['format'] != 'sciona.fitted_sparse_clip.v1'
            or state['feature_columns'] != feature_columns
            or state['feature_coordinate'] != feature_coordinate
            or type(state['upper']) not in (int, float)):
        raise ValueError('Matching numeric clipping state required')
    return FittedSparseClip(len(feature_columns), state['upper']).transform(values)
