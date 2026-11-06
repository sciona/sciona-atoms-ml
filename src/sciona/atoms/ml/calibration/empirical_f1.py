"""Empirical binary F1 thresholds, independent of application and split policy."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete finite scores and observed binary labels required')


def _scores(scores):
    values = np.asarray(scores)
    if values.dtype != np.float64 or values.ndim != 2 or not all(values.shape):
        raise ValueError('Nonempty float64 score matrix required')
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Finite scores in [0,1] required')
    return values


@register_atom(concrete_only)
def fit_empirical_f1_thresholds(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Maximize held-out empirical F1 independently for each score column.

    Nonempty finite float64 scores (N,L) in [0,1] and aligned int64 binary
    labels are required; every column must contain both classes. Predictions
    use score >= threshold. Among equal computed float64 F1 values, choose
    the highest observed-score threshold. Tied scores are processed together.
    Returns a new float64 vector (L,); inputs are unchanged. Sorting takes
    O(L*N*log(N)) time and O(N) temporary memory per column.

    Caller must enforce training/calibration/query separation. This fits a
    decision threshold, not calibrated probabilities or unbiased accuracy.
    No expected-F1 approximation, class weighting or domain policy is used.
    """
    values = _scores(scores)
    targets = np.asarray(labels)
    if targets.dtype != np.int64 or targets.shape != values.shape or not np.isin(targets, (0,1)).all():
        raise ValueError('Aligned int64 binary labels required')
    if not ((targets.min(axis=0) == 0) & (targets.max(axis=0) == 1)).all():
        raise ValueError('Both classes required in each column')
    thresholds = np.empty(values.shape[1], dtype=np.float64)
    for column_index, (column, target) in enumerate(zip(values.T, targets.T, strict=True)):
        order = np.argsort(-column, kind='stable')
        sorted_scores = column[order]
        positives = np.cumsum(target[order], dtype=np.int64)
        total = int(positives[-1])
        ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
        best_score = -1.
        best_threshold = None
        # Descending thresholds and strict improvement retain the highest tie.
        for end in ends:
            f1 = 2 * int(positives[end]) / (int(end) + 1 + total)
            if f1 > best_score:
                best_score = f1
                best_threshold = sorted_scores[end]
        thresholds[column_index] = best_threshold
    return thresholds


@register_atom(concrete_only)
def apply_column_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply inclusive per-column thresholds to finite float64 scores (N,L).

    Scores must be nonempty and in [0,1]. Thresholds are a finite float64 vector
    (L,), allowed outside [0,1] to express all/none decisions. Returns a new
    int64 binary matrix of identical shape. No fitting, clipping or mutation.
    """
    values = _scores(scores)
    cutoffs = np.asarray(thresholds)
    if cutoffs.dtype != np.float64 or cutoffs.shape != (values.shape[1],) or not np.isfinite(cutoffs).all():
        raise ValueError('Aligned finite float64 thresholds required')
    return (values >= cutoffs).astype(np.int64)
