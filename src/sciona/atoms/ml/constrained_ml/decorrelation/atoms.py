"""Decorrelation and fairness-aware evaluation atoms for constrained ML.

Pure functions that measure or enforce statistical independence between
classifier predictions and a protected (or nuisance) variable. Useful for
physics-aware classifiers where the output must not correlate with a
reconstructed mass, or for fairness applications where predictions must
be independent of a sensitive attribute.

All implementations are reimplemented from standard statistical
definitions using numpy and scipy only. The Cramer-von Mises rolling
window approach and the ROC-curve KS trick are well-known techniques
in the HEP-ML community.

Source concepts: Anderson (1962) for the CvM statistic,
Rogozhnikov et al. (hep_ml) for the rolling-window decorrelation
evaluation pattern, LHCb Flavours of Physics Kaggle competition for
the truncated weighted AUC metric.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from scipy.stats import rankdata

from .witnesses import (
    witness_compute_cvm_mass_decorrelation,
    witness_compute_ks_agreement,
    witness_flatness_penalty_gradient,
    witness_noise_injection_decorrelation,
    witness_roc_auc_truncated_weighted,
)


# ---------------------------------------------------------------------------
# Pure-numpy ROC curve (avoids sklearn dependency)
# ---------------------------------------------------------------------------


def _roc_curve_numpy(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute ROC curve from binary labels and continuous scores using numpy only.

    Sorts by descending score, accumulates weighted true-positive and
    false-positive rates at each distinct threshold.

    Args:
        labels: Binary labels (0 or 1), shape (n,).
        scores: Prediction scores, shape (n,).
        sample_weight: Per-sample weights, shape (n,). Uniform if None.

    Returns:
        Tuple of (fpr, tpr, thresholds) arrays.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(scores, dtype=np.float64)

    # Sort by descending score
    desc = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[desc]
    sorted_labels = labels[desc].astype(np.float64)
    sorted_weights = sample_weight[desc]

    # Identify distinct thresholds (keep last occurrence of each score)
    distinct = np.concatenate([
        np.where(np.diff(sorted_scores) != 0)[0],
        np.array([len(sorted_scores) - 1]),
    ])

    # Cumulative weighted TP and FP
    tp_weight = sorted_weights * sorted_labels
    fp_weight = sorted_weights * (1.0 - sorted_labels)
    cum_tp = np.cumsum(tp_weight)[distinct]
    cum_fp = np.cumsum(fp_weight)[distinct]

    total_tp = cum_tp[-1]
    total_fp = cum_fp[-1]

    tpr = cum_tp / total_tp if total_tp > 0 else cum_tp
    fpr = cum_fp / total_fp if total_fp > 0 else cum_fp

    # Prepend (0, 0) origin
    tpr = np.concatenate([np.array([0.0]), tpr])
    fpr = np.concatenate([np.array([0.0]), fpr])
    thresholds = np.concatenate([
        np.array([sorted_scores[distinct[0]] + 1.0]),
        sorted_scores[distinct],
    ])

    return fpr, tpr, thresholds


# ---------------------------------------------------------------------------
# CvM mass decorrelation
# ---------------------------------------------------------------------------


def _cvm_statistic(
    subindices: NDArray[np.int64],
    total_events: int,
) -> float:
    """Compute Cramer-von Mises distance between a subset CDF and uniform CDF.

    Given ranked indices of a subset drawn from a population of size
    total_events, compare the empirical CDF of the subset against
    the global uniform CDF = i/N.

    Args:
        subindices: Rank indices of the subset events within the full population.
        total_events: Total number of events in the population.

    Returns:
        Mean squared difference between the two CDFs.
    """
    target_cdf = np.arange(1, total_events + 1, dtype=np.float64) / total_events
    subset_counts = np.bincount(subindices, minlength=total_events)
    subset_cdf = np.cumsum(subset_counts, dtype=np.float64)
    subset_cdf /= subset_cdf[-1]
    return float(np.mean((target_cdf - subset_cdf) ** 2))


def _rolling_window(
    data: NDArray[np.int64],
    window_size: int,
) -> NDArray[np.int64]:
    """Create a rolling window view over a 1-D array using stride tricks.

    Args:
        data: 1-D input array.
        window_size: Width of each window.

    Returns:
        2-D array of shape (n - window_size + 1, window_size).
    """
    shape = (data.shape[0] - window_size + 1, window_size)
    strides = (data.strides[0], data.strides[0])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


@register_atom(witness_compute_cvm_mass_decorrelation)
@icontract.require(
    lambda predictions, protected_variable: len(predictions) == len(protected_variable),
    "predictions and protected_variable must have the same length",
)
@icontract.require(
    lambda predictions: len(predictions) >= 10,
    "need at least 10 samples for meaningful CvM evaluation",
)
@icontract.require(
    lambda n_neighbours: n_neighbours >= 2,
    "window size must be at least 2",
)
@icontract.require(
    lambda step: step >= 1,
    "step must be at least 1",
)
@icontract.ensure(
    lambda result: np.isfinite(result) and result >= 0.0,
    "CvM value must be a non-negative finite number",
)
def compute_cvm_mass_decorrelation(
    predictions: NDArray[np.float64],
    protected_variable: NDArray[np.float64],
    n_neighbours: int = 200,
    step: int = 50,
) -> float:
    """Measure prediction-mass decorrelation via the Cramer-von Mises statistic.

    Sorts events by the protected variable (e.g. reconstructed mass),
    then slides a window of n_neighbours events along the sorted axis.
    Within each window, the CDF of ranked predictions is compared to
    the global prediction CDF using the Cramer-von Mises distance.
    The returned value is the average CvM across all windows.

    Lower values indicate that the classifier output is more
    independent of the protected variable. A perfectly decorrelated
    classifier returns a value near zero.

    The rolling-window CvM approach avoids binning the protected
    variable and instead uses a smooth sliding neighbourhood,
    making it robust to the choice of bin edges.

    Args:
        predictions: Classifier output scores, shape (n,).
        protected_variable: Continuous protected variable (e.g. mass), shape (n,).
        n_neighbours: Number of neighbouring events in each local window.
        step: Step size for sliding the window center through the sorted array.

    Returns:
        Average CvM statistic across all windows. Lower is better.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    protected_variable = np.asarray(protected_variable, dtype=np.float64)

    n = len(predictions)
    effective_neighbours = min(n_neighbours, n)

    # Sort predictions by the protected variable
    sort_order = np.argsort(protected_variable)
    sorted_preds = predictions[sort_order]

    # Replace predictions with their global rank
    ranked = np.argsort(np.argsort(sorted_preds, kind="mergesort"), kind="mergesort")

    # Compute CvM for each rolling window
    windows = _rolling_window(ranked, effective_neighbours)
    cvms = []
    for window in windows[::step]:
        cvms.append(_cvm_statistic(window, total_events=n))

    return float(np.mean(cvms))


# ---------------------------------------------------------------------------
# KS agreement
# ---------------------------------------------------------------------------


@register_atom(witness_compute_ks_agreement)
@icontract.require(
    lambda data_predictions, weights_data: len(data_predictions) == len(weights_data),
    "data_predictions and weights_data must have the same length",
)
@icontract.require(
    lambda mc_predictions, weights_mc: len(mc_predictions) == len(weights_mc),
    "mc_predictions and weights_mc must have the same length",
)
@icontract.require(
    lambda data_predictions: len(data_predictions) >= 1,
    "need at least 1 data sample",
)
@icontract.require(
    lambda mc_predictions: len(mc_predictions) >= 1,
    "need at least 1 MC sample",
)
@icontract.ensure(
    lambda result: np.isfinite(result) and 0.0 <= result <= 1.0,
    "KS distance must be in [0, 1]",
)
def compute_ks_agreement(
    data_predictions: NDArray[np.float64],
    mc_predictions: NDArray[np.float64],
    weights_data: NDArray[np.float64],
    weights_mc: NDArray[np.float64],
) -> float:
    """Compute the Kolmogorov-Smirnov distance between two weighted prediction distributions.

    Measures how well the prediction distribution on one dataset (e.g.
    real data) agrees with that on another (e.g. Monte Carlo simulation).
    Uses the ROC-curve identity: the max absolute difference between FPR
    and TPR on a binary classification task (label 0 = dataset A,
    label 1 = dataset B) equals the KS distance between the two
    weighted score distributions.

    Lower values indicate better agreement between the two distributions.

    Args:
        data_predictions: Prediction scores from dataset A, shape (n_data,).
        mc_predictions: Prediction scores from dataset B, shape (n_mc,).
        weights_data: Sample weights for dataset A, shape (n_data,).
        weights_mc: Sample weights for dataset B, shape (n_mc,).

    Returns:
        KS distance in [0, 1]. 0 means identical distributions.
    """
    data_predictions = np.asarray(data_predictions, dtype=np.float64)
    mc_predictions = np.asarray(mc_predictions, dtype=np.float64)
    weights_data = np.asarray(weights_data, dtype=np.float64)
    weights_mc = np.asarray(weights_mc, dtype=np.float64)

    # Normalize weights
    w_data = weights_data / np.sum(weights_data)
    w_mc = weights_mc / np.sum(weights_mc)

    # Build binary classification problem: 0=data, 1=mc
    labels = np.concatenate([
        np.zeros(len(data_predictions), dtype=np.int64),
        np.ones(len(mc_predictions), dtype=np.int64),
    ])
    scores = np.concatenate([data_predictions, mc_predictions])
    weights = np.concatenate([w_data, w_mc])

    fpr, tpr, _ = _roc_curve_numpy(labels, scores, sample_weight=weights)
    return float(np.max(np.abs(fpr - tpr)))


# ---------------------------------------------------------------------------
# Truncated weighted AUC
# ---------------------------------------------------------------------------


@register_atom(witness_roc_auc_truncated_weighted)
@icontract.require(
    lambda labels, predictions: len(labels) == len(predictions),
    "labels and predictions must have the same length",
)
@icontract.require(
    lambda labels: len(labels) >= 2,
    "need at least 2 samples",
)
@icontract.require(
    lambda tpr_thresholds, weights: len(tpr_thresholds) + 1 == len(weights),
    "weights must have exactly one more element than tpr_thresholds",
)
@icontract.ensure(
    lambda result: np.isfinite(result) and 0.0 <= result <= 1.0,
    "weighted AUC must be in [0, 1]",
)
def roc_auc_truncated_weighted(
    labels: NDArray[np.float64],
    predictions: NDArray[np.float64],
    tpr_thresholds: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    weights: tuple[float, ...] = (4, 3, 2, 1, 0),
) -> float:
    """Compute a weighted area under the ROC curve segmented by TPR thresholds.

    Partitions the ROC curve into segments delimited by TPR thresholds and
    computes a weighted sum of the areas under each segment. This allows
    emphasizing specific operating regions of the classifier, such as the
    high-precision (low TPR) region where false positives are expensive.

    With the default weights (4, 3, 2, 1, 0), only the region where
    TPR < 0.8 contributes, with the strongest weight on the region
    TPR < 0.2. The result is normalized so that a perfect classifier
    achieves a score of 1.0.

    Args:
        labels: Binary ground truth labels (0 or 1), shape (n,).
        predictions: Classifier prediction scores in [0, 1], shape (n,).
        tpr_thresholds: Boundaries between ROC segments, in ascending order.
        weights: Weight for each segment. Length must be len(tpr_thresholds) + 1.

    Returns:
        Weighted AUC in [0, 1]. Higher is better.
    """
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)

    fpr, tpr, _ = _roc_curve_numpy(
        labels.astype(np.int64), predictions,
    )

    # Build full boundary list: [0, threshold_1, ..., threshold_k, 1]
    boundaries = [0.0] + list(tpr_thresholds) + [1.0]

    area = 0.0
    for i in range(1, len(boundaries)):
        # Clip TPR to this segment's upper and lower boundaries
        tpr_upper = np.minimum(tpr, boundaries[i])
        tpr_lower = np.minimum(tpr, boundaries[i - 1])
        # Segment area = integral of (clipped_upper - clipped_lower) over FPR
        segment_area = np.trapezoid(tpr_upper - tpr_lower, fpr)
        area += weights[i - 1] * segment_area

    # Normalize: ideal classifier has rectangular segments
    boundaries_arr = np.array(boundaries)
    weights_arr = np.array(weights, dtype=np.float64)
    normalization = float(np.sum(
        (boundaries_arr[1:] - boundaries_arr[:-1]) * weights_arr
    ))

    if normalization < 1e-15:
        return 0.0

    return float(area / normalization)


# ---------------------------------------------------------------------------
# Noise injection decorrelation
# ---------------------------------------------------------------------------


@register_atom(witness_noise_injection_decorrelation)
@icontract.require(
    lambda predictions: len(predictions) >= 1,
    "need at least 1 prediction",
)
@icontract.require(
    lambda noise_level: 0.0 <= noise_level <= 1.0,
    "noise_level must be in [0, 1]",
)
@icontract.ensure(
    lambda result: len(result) >= 1,
    "output must have at least 1 element",
)
def noise_injection_decorrelation(
    predictions: NDArray[np.float64],
    noise_level: float = 0.40,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Decorrelate predictions by mixing with uniform random noise.

    Applies the linear interpolation: output = level * U + (1 - level) * pred,
    where U is drawn from Uniform(0, 1). This is a brute-force but effective
    method to reduce correlation between predictions and any nuisance variable.
    Higher noise levels reduce correlation more aggressively but degrade
    discriminating power.

    This is commonly used as a baseline decorrelation strategy before
    evaluating more sophisticated methods like adversarial training or
    flatness-constrained boosting.

    Args:
        predictions: Classifier output scores, shape (n,).
        noise_level: Fraction of uniform noise to mix in (0 = no noise, 1 = pure noise).
        random_state: Seed for reproducibility. None uses non-deterministic randomness.

    Returns:
        Decorrelated predictions, shape (n,). Same length as input.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    noise = rng.uniform(0.0, 1.0, size=len(predictions))
    return noise_level * noise + (1.0 - noise_level) * predictions


@register_atom(witness_flatness_penalty_gradient)
@icontract.require(lambda predictions: predictions.ndim == 1, "predictions must be 1-D")
@icontract.require(lambda labels: np.all((labels == 0) | (labels == 1)), "labels must be binary {0, 1}")
@icontract.require(
    lambda predictions, labels, protected_variable: (
        len(predictions) == len(labels) == len(protected_variable)
    ),
    "all arrays must have the same length",
)
@icontract.require(lambda n_bins: n_bins >= 2, "need at least 2 bins")
@icontract.require(lambda power: power >= 1.0, "power must be >= 1")
@icontract.require(lambda fl_coefficient: fl_coefficient >= 0.0, "fl_coefficient must be non-negative")
@icontract.ensure(lambda result, predictions: result.shape == predictions.shape, "output shape matches input")
def flatness_penalty_gradient(
    predictions: NDArray[np.float64],
    labels: NDArray[np.int64],
    protected_variable: NDArray[np.float64],
    n_bins: int = 10,
    power: float = 2.0,
    fl_coefficient: float = 3.0,
    uniform_label: int = 0,
) -> NDArray[np.float64]:
    """Compute the negative gradient of the flatness-penalized boosting loss.

    Combines an AdaBoost-style exponential loss for classification quality
    with a binned CDF-mismatch penalty that encourages predictions to be
    statistically independent of a protected (nuisance) variable.

    The flatness penalty compares the empirical CDF of predictions within
    each bin of the protected variable against the global CDF. The gradient
    pushes per-bin CDFs toward the global CDF, enforcing uniformity.

    This is the core loss function used in flatness-constrained gradient
    boosting (Rogozhnikov et al., arXiv:1410.4140). It can be passed as
    a custom objective to boosting libraries that accept callable losses
    (e.g., XGBoost ``obj`` parameter).

    Args:
        predictions: Current ensemble predictions, shape (n,).
        labels: Binary class labels {0, 1}, shape (n,).
        protected_variable: Nuisance variable to decorrelate from (e.g., mass), shape (n,).
        n_bins: Number of quantile bins along the protected variable.
        power: Exponent for the flatness penalty (2.0 = quadratic CDF mismatch).
        fl_coefficient: Weight of the flatness penalty relative to the
            classification loss (lambda in L = L_ada + lambda * L_flat).
        uniform_label: Which class label's predictions must be flat with
            respect to the protected variable (typically 0 = background).

    Returns:
        Negative gradient (pseudo-residuals), shape (n,). Suitable as the
        residuals for the next boosting tree.

    References:
        Rogozhnikov et al. (2015), "New approaches for boosting to
        uniformity", arXiv:1410.4140. Implementation concept from hep_ml
        (Apache 2.0).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    protected_variable = np.asarray(protected_variable, dtype=np.float64)
    n = len(predictions)

    # --- Classification gradient (AdaBoost exponential loss) ---
    y_signed = 2 * labels - 1  # {0,1} -> {-1,+1}
    margin = y_signed * predictions
    ada_grad = y_signed * np.exp(np.clip(-margin, -1e5, 2.0))

    # --- Flatness gradient (binned CDF mismatch) ---
    flatness_grad = np.zeros(n, dtype=np.float64)

    # Only constrain the specified label
    constraint_mask = labels == uniform_label
    if np.sum(constraint_mask) < 2 * n_bins:
        # Not enough samples to bin — skip flatness term
        return ada_grad

    # Bin the protected variable using quantiles (constraint samples only)
    constrained_preds = predictions[constraint_mask]
    constrained_prot = protected_variable[constraint_mask]
    n_c = len(constrained_preds)

    bin_edges = np.percentile(constrained_prot, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10  # ensure max value falls in last bin
    bin_indices = np.digitize(constrained_prot, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Global ranks (normalized to [0, 1])
    global_rank = rankdata(constrained_preds, method="average") / n_c

    # Per-bin flatness gradient
    flat_grad_constrained = np.zeros(n_c, dtype=np.float64)
    for b in range(n_bins):
        mask_b = bin_indices == b
        n_b = np.sum(mask_b)
        if n_b < 2:
            continue

        # Ranks within this bin (normalized to [0, 1])
        local_rank = rankdata(constrained_preds[mask_b], method="average") / n_b
        global_pos = global_rank[mask_b]

        diff = local_rank - global_pos
        grad_mag = power * np.abs(diff) ** (power - 1)
        flat_grad_constrained[mask_b] = np.sign(diff) * grad_mag

    flatness_grad[constraint_mask] = flat_grad_constrained

    # --- Combine ---
    return ada_grad + fl_coefficient * flatness_grad
