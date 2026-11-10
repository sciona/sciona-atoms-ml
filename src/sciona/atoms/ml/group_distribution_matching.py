"""Score explicit whole-group subsets through the existing KS-distance atom."""
import numpy as np
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from sciona.atoms.ml.constrained_ml.decorrelation.atoms import compute_ks_agreement


def _layout(training, reference, row_groups, candidates, group_labels, feature_columns, feature_coordinate):
    if (len(training.shape) != 2 or len(reference.shape) != 2
            or min(training.shape) < 1 or min(reference.shape) < 1
            or training.shape[1] != reference.shape[1]
            or np.dtype(training.dtype).kind not in 'iuf' or np.dtype(reference.dtype).kind not in 'iuf'):
        raise ValueError('Nonempty aligned real feature matrices required')
    if row_groups.shape != (training.shape[0],) or np.dtype(row_groups.dtype).kind not in 'iu':
        raise ValueError('Integer group label for each training row required')
    if (type(group_labels) is not list or not group_labels
            or any(type(g) is not int for g in group_labels) or len(set(group_labels)) != len(group_labels)):
        raise ValueError('Unique explicit integer group labels required')
    if (len(candidates.shape) != 2 or candidates.shape[0] < 1
            or candidates.shape[1] != len(group_labels) or np.dtype(candidates.dtype).kind != 'b'):
        raise ValueError('Boolean candidates-by-groups mask required')
    if (type(feature_columns) is not list or len(feature_columns) != training.shape[1]
            or any(type(c) is not str or not c.strip() for c in feature_columns)
            or len(set(feature_columns)) != len(feature_columns)
            or type(feature_coordinate) is not str or not feature_coordinate.strip()):
        raise ValueError('Explicit unique feature order and shared coordinate required')


def witness_group_subset_ks(training: AbstractArray, reference: AbstractArray,
        row_groups: AbstractArray, candidates: AbstractArray, group_labels: list,
        feature_columns: list, feature_coordinate: str) -> AbstractArray:
    _layout(training, reference, row_groups, candidates, group_labels, feature_columns, feature_coordinate)
    if training.dim != reference.dim:
        raise ValueError('Feature dimensions must agree')
    return AbstractArray(shape=(candidates.shape[0], training.shape[1]), dtype='float64', min_val=0., max_val=1.)


@register_atom(witness_group_subset_ks)
def group_subset_ks(training: np.ndarray, reference: np.ndarray, row_groups: np.ndarray,
        candidates: np.ndarray, group_labels: list, feature_columns: list,
        feature_coordinate: str) -> np.ndarray:
    """Compare each supplied whole-group subset with a reference population.

    Returns candidates-by-features two-sided empirical KS distances, using
    the registered compute_ks_agreement operation with equal row weights.
    Groups contribute all their rows, so larger groups carry more mass. Group
    labels must exactly cover observed training labels; every candidate selects
    at least one group. Duplicate candidates are permitted and retain ordering.
    Feature columns and their coordinate must be aligned by the caller across
    both matrices. No scaling, feature screening, random subset generation,
    p-values, averaging, tie-breaking or automatic winner selection occurs.
    Inputs must be finite and float64-representable; very large integers may
    lose precision. Inputs are not mutated. Executes serially without workers.
    Using a future/query reference population makes selection transductive;
    it is unsuitable where that population is unavailable at decision time.
    Small distances do not establish independence, predictive quality, joint
    distribution agreement or a valid causal/temporal validation split.
    Witnesses check layout only; runtime additionally checks values and groups.
    """
    if any(not isinstance(x, np.ndarray) for x in (training, reference, row_groups, candidates)):
        raise ValueError('NumPy arrays required')
    _layout(training, reference, row_groups, candidates, group_labels, feature_columns, feature_coordinate)
    if set(map(int, row_groups)) != set(group_labels) or not candidates.any(axis=1).all():
        raise ValueError('Exact observed group coverage and nonempty subsets required')
    with np.errstate(over='ignore', invalid='ignore'):
        X, R = training.astype(np.float64), reference.astype(np.float64)
    if not np.isfinite(X).all() or not np.isfinite(R).all():
        raise ValueError('Finite float64-representable features required')
    result = np.empty((len(candidates), X.shape[1]), dtype=np.float64)
    reference_weights = np.ones(len(R))
    for i, mask in enumerate(candidates):
        selected = [label for label, use in zip(group_labels, mask) if use]
        subset = X[np.isin(row_groups, selected)]
        weights = np.ones(len(subset))
        for j in range(X.shape[1]):
            result[i, j] = compute_ks_agreement(subset[:, j], R[:, j], weights, reference_weights)
    return result
