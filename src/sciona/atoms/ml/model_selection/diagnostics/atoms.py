"""Dataset diagnostic atoms for deterministic ML model selection.

Pure functions that compute a single statistic from the design matrix X,
target vector y, or both. These diagnostics are consumed by the
recommendation atoms in the sibling family to produce model selection
guidance.

Each atom cites the specific threshold and source from the Gemini deep
research survey (heuristics.pdf) that motivates its inclusion.

Source: Decision Heuristics for Deterministic Model Selection in
scikit-learn (heuristics.pdf), derived from primary statistical and
ML theory sources.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy import sparse

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_check_lasso_sample_complexity,
    witness_check_time_series_index,
    witness_compute_condition_number,
    witness_compute_dispersion_index,
    witness_compute_excess_kurtosis,
    witness_compute_explained_variance_ratio,
    witness_compute_mutual_incoherence,
    witness_compute_n_p_ratio,
    witness_compute_residual_kurtosis,
    witness_compute_skewness,
    witness_compute_vif,
    witness_count_categorical_features,
    witness_estimate_noise_level,
    witness_estimate_tweedie_power,
    witness_is_sparse,
    witness_test_normality,
)


# ---------------------------------------------------------------------------
# Section A: Regularization diagnostics
# ---------------------------------------------------------------------------


@register_atom(witness_compute_condition_number)
@icontract.require(lambda X: X.shape[0] >= 1 and X.shape[1] >= 1, "X must be non-empty")
@icontract.ensure(lambda result: np.isfinite(result) and result >= 1.0, "condition number must be >= 1")
def compute_condition_number(X: NDArray[np.float64]) -> float:
    """Compute the condition number kappa(X) = sigma_max / sigma_min.

    Measures ill-conditioning of the design matrix. When kappa(X) > 30,
    OLS becomes unstable and regularization is required. The condition
    number bounds the relative error amplification in the solution of
    the linear system under statistical perturbations.

    Threshold: kappa(X) > 30 (OLS Instability Limit)
    Source: Hoerl & Kennard (1970); heuristics.pdf Section A

    Args:
        X: Design matrix, shape (n, p).

    Returns:
        Condition number (ratio of largest to smallest singular value).
        Returns 1e12 if the matrix is numerically singular.
    """
    s = np.linalg.svd(X, compute_uv=False)
    if s[-1] < 1e-15:
        return 1e12
    return float(s[0] / s[-1])


@register_atom(witness_compute_n_p_ratio)
@icontract.require(lambda X: X.shape[1] >= 1, "X must have at least one feature")
@icontract.ensure(lambda result: result > 0.0, "ratio must be positive")
def compute_n_p_ratio(X: NDArray[np.float64]) -> float:
    """Compute the sample-to-feature ratio n/p.

    A fundamental quantity for model selection. When n/p < 1, the system
    is underdetermined and regularization is mandatory. When n/p >= 10,
    OLS is typically viable if the condition number is low.

    Threshold: n/p >= 10 for OLS viability
    Source: heuristics.pdf Section A

    Args:
        X: Design matrix, shape (n, p).

    Returns:
        Ratio n/p as a float.
    """
    return float(X.shape[0] / X.shape[1])


@register_atom(witness_compute_mutual_incoherence)
@icontract.require(lambda X: X.shape[1] >= 2, "X must have at least 2 features")
@icontract.ensure(lambda result: np.isfinite(result) and result >= 0.0, "incoherence must be non-negative")
def compute_mutual_incoherence(X: NDArray[np.float64]) -> float:
    """Compute the mutual incoherence mu = max_{i != j} |X_i^T X_j|.

    Measures the maximum absolute cross-correlation between any two
    columns of the normalized design matrix. When mu <= 1, Lasso can
    provably recover the true sparse support. When mu > 1, Lasso's
    variable selection consistency breaks down (irrepresentable condition
    violation), and ElasticNet or Ridge should be preferred.

    Threshold: mu <= 1 for Lasso recovery guarantee
    Source: Wainwright (2009); heuristics.pdf Section A

    Args:
        X: Design matrix, shape (n, p). Columns are normalized internally.

    Returns:
        Mutual incoherence value. 0 = perfectly incoherent (ideal for
        Lasso), > 1 = highly coherent (Lasso unreliable).
    """
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    X_normed = X / norms
    gram = X_normed.T @ X_normed
    np.fill_diagonal(gram, 0.0)
    return float(np.max(np.abs(gram)))


@register_atom(witness_check_lasso_sample_complexity)
@icontract.require(lambda X: X.shape[0] >= 1, "X must be non-empty")
@icontract.require(lambda sparsity_estimate: sparsity_estimate >= 1, "sparsity must be at least 1")
@icontract.ensure(lambda result: isinstance(result, bool), "must return bool")
def check_lasso_sample_complexity(
    X: NDArray[np.float64],
    sparsity_estimate: int,
) -> bool:
    """Check whether sample size satisfies the Lasso recovery bound n > 2k log(p-k).

    For a Gaussian ensemble, if n drops below 2k*log(p-k), the probability
    of successful sparsity pattern recovery by L1 converges to zero. This
    is a necessary (not sufficient) condition for Lasso to work.

    Threshold: n > 2k * log(p - k) for sharp recovery
    Source: Wainwright (2009); heuristics.pdf Section A

    Args:
        X: Design matrix, shape (n, p).
        sparsity_estimate: Estimated number of truly non-zero coefficients k.

    Returns:
        True if the sample complexity bound is satisfied.
    """
    n, p = X.shape
    k = min(sparsity_estimate, p - 1)
    if p - k <= 0:
        return True
    threshold = 2 * k * np.log(p - k)
    return bool(n > threshold)


# ---------------------------------------------------------------------------
# Section B: Loss function diagnostics
# ---------------------------------------------------------------------------


@register_atom(witness_compute_excess_kurtosis)
@icontract.require(lambda y: len(y) >= 4, "need at least 4 samples for kurtosis")
@icontract.ensure(lambda result: np.isfinite(result), "kurtosis must be finite")
def compute_excess_kurtosis(y: NDArray[np.float64]) -> float:
    """Compute the excess kurtosis of the target variable.

    Excess kurtosis measures tail heaviness relative to a Gaussian
    (which has excess kurtosis = 0). When excess kurtosis > 1.0, OLS
    estimates become highly sensitive to outliers and robust loss
    functions (Huber, MAE) should be preferred.

    Threshold: excess kurtosis > 1.0 for robust loss
    Source: heuristics.pdf Section B

    Args:
        y: Target variable, shape (n,).

    Returns:
        Fisher excess kurtosis (kurtosis - 3).
    """
    return float(stats.kurtosis(y, fisher=True))


@register_atom(witness_compute_residual_kurtosis)
@icontract.require(lambda X, y: X.shape[0] == len(y), "X and y must have matching samples")
@icontract.require(lambda X: X.shape[0] > X.shape[1], "need n > p for OLS residuals")
@icontract.ensure(lambda result: np.isfinite(result), "residual kurtosis must be finite")
def compute_residual_kurtosis(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Compute excess kurtosis of OLS residuals to detect outlier contamination.

    Fits a fast baseline OLS model and measures the tail heaviness of the
    residuals. This is more informative than target kurtosis alone because
    it isolates the noise distribution from the signal.

    Threshold: residual excess kurtosis > 1.0 for robust loss
    Source: heuristics.pdf Section B

    Args:
        X: Design matrix, shape (n, p).
        y: Target variable, shape (n,).

    Returns:
        Excess kurtosis of OLS residuals.
    """
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coeffs
    return float(stats.kurtosis(residuals, fisher=True))


# ---------------------------------------------------------------------------
# Section C: Linear model family diagnostics
# ---------------------------------------------------------------------------


@register_atom(witness_compute_dispersion_index)
@icontract.require(lambda y: len(y) >= 2, "need at least 2 samples")
@icontract.require(lambda y: np.mean(y) > 0, "mean must be positive for dispersion index")
@icontract.ensure(lambda result: np.isfinite(result) and result >= 0.0, "dispersion index must be non-negative")
def compute_dispersion_index(y: NDArray[np.float64]) -> float:
    """Compute the dispersion index DI = Var(y) / E(y).

    The primary metric for evaluating Poisson model validity for count
    data. DI = 1 implies Poisson, DI > 1.1 indicates overdispersion
    requiring Tweedie or Negative Binomial.

    Threshold: DI ~= 1 for Poisson, DI > 1.1 for overdispersion
    Source: heuristics.pdf Section C

    Args:
        y: Target variable (must be non-negative), shape (n,).

    Returns:
        Dispersion index (variance / mean).
    """
    return float(np.var(y) / np.mean(y))


@register_atom(witness_estimate_tweedie_power)
@icontract.require(lambda y: len(y) >= 10, "need at least 10 samples")
@icontract.require(lambda y: np.all(y >= 0), "y must be non-negative for Tweedie")
@icontract.ensure(lambda result: np.isfinite(result), "power must be finite")
def estimate_tweedie_power(y: NDArray[np.float64]) -> float:
    """Estimate the Tweedie power parameter p where Var(y) ~ E(y)^p.

    Uses binned estimation: sorts y, groups into quantile bins, computes
    mean and variance per bin, then fits log(Var) ~ p * log(Mean) via
    OLS. p=1 is Poisson, p=2 is Gamma, 1<p<2 is compound Poisson-Gamma.

    Source: heuristics.pdf Section C

    Args:
        y: Non-negative target variable, shape (n,).

    Returns:
        Estimated Tweedie power parameter.
    """
    n = len(y)
    n_bins = max(5, min(20, n // 20))
    sorted_y = np.sort(y)
    bins = np.array_split(sorted_y, n_bins)

    means = []
    variances = []
    for b in bins:
        m = np.mean(b)
        v = np.var(b)
        if m > 1e-10 and v > 1e-10:
            means.append(m)
            variances.append(v)

    if len(means) < 3:
        return 1.0  # default to Poisson

    log_means = np.log(np.array(means))
    log_vars = np.log(np.array(variances))
    coeffs = np.polyfit(log_means, log_vars, 1)
    return float(np.clip(coeffs[0], 0.0, 3.0))


# ---------------------------------------------------------------------------
# Section D: Tree ensemble diagnostics
# ---------------------------------------------------------------------------


@register_atom(witness_estimate_noise_level)
@icontract.require(lambda X, y: X.shape[0] == len(y), "X and y must have matching samples")
@icontract.require(lambda X: X.shape[0] >= 10, "need at least 10 samples")
@icontract.ensure(lambda result: np.isfinite(result) and result >= 0.0, "noise must be non-negative")
def estimate_noise_level(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Estimate irreducible noise by fitting a shallow decision tree.

    Fits a DecisionTreeRegressor(max_depth=3) and returns the variance
    of residuals. High noise favors bagging (RandomForest) over boosting
    (GradientBoosting), since boosting is more susceptible to memorizing
    noise.

    Source: heuristics.pdf Section D

    Args:
        X: Design matrix, shape (n, p).
        y: Target variable, shape (n,).

    Returns:
        Estimated noise variance (residual variance from shallow tree).
    """
    from sklearn.tree import DecisionTreeRegressor

    tree = DecisionTreeRegressor(max_depth=3, random_state=0)
    tree.fit(X, y)
    residuals = y - tree.predict(X)
    return float(np.var(residuals))


@register_atom(witness_count_categorical_features)
@icontract.require(lambda X: X.ndim == 2, "X must be 2-dimensional")
@icontract.ensure(lambda result: result >= 0, "count must be non-negative")
def count_categorical_features(X: NDArray[np.float64]) -> int:
    """Count features that appear to be unencoded categoricals.

    A feature is considered categorical if the number of unique values
    is less than 20 and all values are integers (or near-integer).
    HistGradientBoosting natively handles categoricals; other models
    require encoding.

    Source: heuristics.pdf Section D

    Args:
        X: Design matrix, shape (n, p).

    Returns:
        Number of columns that appear categorical.
    """
    count = 0
    for j in range(X.shape[1]):
        col = X[:, j]
        unique_count = len(np.unique(col))
        if unique_count < 20 and np.allclose(col, np.round(col)):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Section E: Preprocessing diagnostics
# ---------------------------------------------------------------------------


@register_atom(witness_compute_skewness)
@icontract.require(lambda x: len(x) >= 3, "need at least 3 samples")
@icontract.ensure(lambda result: np.isfinite(result), "skewness must be finite")
def compute_skewness(x: NDArray[np.float64]) -> float:
    """Compute the Fisher-Pearson skewness of a single feature.

    Skewness outside [-1, 1] indicates high asymmetry that violates
    normality assumptions in gradient descent and distance metrics.
    Features exceeding this threshold should be power-transformed.

    Threshold: |skewness| > 1.0 for PowerTransformer
    Source: heuristics.pdf Section E

    Args:
        x: Single feature column, shape (n,).

    Returns:
        Fisher-Pearson skewness coefficient.
    """
    return float(stats.skew(x))


@register_atom(witness_compute_vif)
@icontract.require(lambda X: X.shape[1] >= 2, "need at least 2 features for VIF")
@icontract.require(lambda X: X.shape[0] > X.shape[1], "need n > p for VIF regression")
@icontract.ensure(lambda result: np.all(result >= 1.0), "VIF must be >= 1")
def compute_vif(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the Variance Inflation Factor for each feature.

    VIF_j = 1 / (1 - R_j^2) where R_j^2 is the R-squared from
    regressing feature j on all other features. VIF > 5 indicates
    moderate collinearity; VIF > 10 indicates severe collinearity
    requiring feature removal or PCA.

    Threshold: VIF > 5 moderate, VIF > 10 severe
    Source: heuristics.pdf Section E

    Args:
        X: Design matrix, shape (n, p).

    Returns:
        VIF values, shape (p,). One per feature.
    """
    p = X.shape[1]
    vif = np.zeros(p)
    for j in range(p):
        mask = np.ones(p, dtype=bool)
        mask[j] = False
        X_other = X[:, mask]
        y_j = X[:, j]
        coeffs, residuals, _, _ = np.linalg.lstsq(X_other, y_j, rcond=None)
        y_hat = X_other @ coeffs
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
        if ss_tot < 1e-15:
            vif[j] = 1.0
        else:
            r_squared = 1.0 - ss_res / ss_tot
            if r_squared >= 1.0:
                vif[j] = 1e6
            else:
                vif[j] = 1.0 / (1.0 - r_squared)
    return vif


@register_atom(witness_test_normality)
@icontract.require(lambda x: 3 <= len(x) <= 5000, "Shapiro-Wilk requires 3-5000 samples")
@icontract.ensure(lambda result: 0.0 <= result <= 1.0, "p-value must be in [0, 1]")
def test_normality(x: NDArray[np.float64]) -> float:
    """Test whether a feature follows a Gaussian distribution (Shapiro-Wilk).

    Low p-values (< 0.05) indicate significant deviation from normality,
    suggesting PowerTransformer or QuantileTransformer should be applied
    before models that assume Gaussian features.

    Source: heuristics.pdf Section E

    Args:
        x: Single feature column, shape (n,). Max 5000 samples for
            Shapiro-Wilk.

    Returns:
        p-value from the Shapiro-Wilk test. Low = non-Gaussian.
    """
    _, p_value = stats.shapiro(x)
    return float(p_value)


@register_atom(witness_is_sparse)
@icontract.require(lambda X: sparse.issparse(X) or isinstance(X, np.ndarray), "X must be dense ndarray or scipy sparse")
@icontract.ensure(lambda result: isinstance(result, bool), "must return bool")
def is_sparse(X: object) -> bool:
    """Check whether the design matrix is stored in a sparse format.

    Sparse matrices require different preprocessing (no mean centering)
    and different decomposition (TruncatedSVD instead of PCA). Also
    affects solver selection for linear models.

    Source: heuristics.pdf Sections C, E, F

    Args:
        X: Design matrix (dense ndarray or scipy sparse matrix).

    Returns:
        True if X is a scipy sparse matrix.
    """
    return bool(sparse.issparse(X))


# ---------------------------------------------------------------------------
# Section F: Dimensionality reduction diagnostics
# ---------------------------------------------------------------------------


@register_atom(witness_compute_explained_variance_ratio)
@icontract.require(lambda X: X.shape[0] >= 2 and X.shape[1] >= 1, "X must be non-empty")
@icontract.require(lambda n_components: n_components >= 1, "need at least 1 component")
@icontract.ensure(lambda result: 0.0 <= result <= 1.0, "ratio must be in [0, 1]")
def compute_explained_variance_ratio(
    X: NDArray[np.float64],
    n_components: int,
) -> float:
    """Compute the fraction of total variance explained by the top n_components.

    This is the cumulative sum of the top eigenvalues of the covariance
    matrix divided by the trace. Standard thresholds: 0.90, 0.95, 0.99.

    Threshold: 0.95 default for PCA retention
    Source: heuristics.pdf Section F

    Args:
        X: Design matrix, shape (n, p).
        n_components: Number of principal components to consider.

    Returns:
        Fraction of total variance explained (0 to 1).
    """
    n_comp = min(n_components, X.shape[1], X.shape[0])
    X_centered = X - np.mean(X, axis=0)
    s = np.linalg.svd(X_centered, compute_uv=False)
    total_variance = np.sum(s ** 2)
    if total_variance < 1e-15:
        return 1.0
    explained = np.sum(s[:n_comp] ** 2)
    return float(explained / total_variance)


# ---------------------------------------------------------------------------
# Section H: Cross-validation diagnostics
# ---------------------------------------------------------------------------


@register_atom(witness_check_time_series_index)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.ensure(lambda result: isinstance(result, bool), "must return bool")
def check_time_series_index(X: NDArray[np.float64]) -> bool:
    """Check whether the first column appears to be a monotonic time index.

    If the data has temporal structure, standard KFold will leak future
    information into training folds. TimeSeriesSplit must be used instead.

    Source: heuristics.pdf Section H

    Args:
        X: Design matrix, shape (n, p). Checks the first column.

    Returns:
        True if the first column is strictly monotonically increasing.
    """
    if X.shape[1] < 1 or X.shape[0] < 2:
        return False
    first_col = X[:, 0]
    return bool(np.all(np.diff(first_col) > 0))
