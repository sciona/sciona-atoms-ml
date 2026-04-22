"""Deterministic model selection recommendation atoms.

Pure functions that consume scalar diagnostics from the sibling diagnostics
family and emit structured recommendation dicts. Each atom encodes one
decision table from the Gemini deep research survey (heuristics.pdf) as
deterministic if/elif logic with named threshold constants.

Every recommendation dict contains:
    recommendation  - short human-readable label
    confidence      - "high", "medium", or "absolute"
    sklearn_class   - fully-qualified sklearn estimator or splitter name
    reasoning       - one-sentence rationale citing the threshold that fired
    alternatives    - list of runner-up sklearn classes
    thresholds_applied - dict of {constant_name: value} that drove the decision
    source_sections - list of heuristics.pdf section letters

Source: Decision Heuristics for Deterministic Model Selection in
scikit-learn (heuristics.pdf), derived from primary statistical and
ML theory sources.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_recommend_regularization,
    witness_recommend_loss_function,
    witness_recommend_linear_model,
    witness_recommend_tree_ensemble,
    witness_recommend_preprocessing,
    witness_recommend_dimensionality_reduction,
    witness_recommend_hyperparameter_ranges,
    witness_recommend_cv_strategy,
)

# ---------------------------------------------------------------------------
# Named threshold constants (heuristics.pdf)
# ---------------------------------------------------------------------------

OLS_INSTABILITY_LIMIT = 30
LASSO_MUTUAL_INCOHERENCE_LIMIT = 1.0
KURTOSIS_ROBUST_LOSS_THRESHOLD = 1.0
DISPERSION_INDEX_POISSON = 1.0
DISPERSION_OVERDISPERSION = 1.1
SKEWNESS_TRANSFORM_THRESHOLD = 1.0
VIF_MODERATE_COLLINEARITY = 5
VIF_SEVERE_COLLINEARITY = 10
HIST_GB_SAMPLE_BOUNDARY = 10_000
EXPLAINED_VARIANCE_DEFAULT = 0.95
SOLVER_FEATURE_LIMIT = 1000
BOOSTING_DIMINISHING_LR = 0.125
BOOSTING_FLATLINE_ESTIMATORS = 200
LOOCV_VIABILITY_BOUNDARY = 100
ALPHA_FLOOR = 1e-5

# Derived thresholds
N_P_RATIO_OLS_VIABLE = 10


# ---------------------------------------------------------------------------
# Section A: Regularization recommendation
# ---------------------------------------------------------------------------


@register_atom(witness_recommend_regularization)
@icontract.require(
    lambda condition_number: condition_number >= 1.0,
    "condition number must be >= 1",
)
@icontract.require(
    lambda n_p_ratio: n_p_ratio > 0.0,
    "n/p ratio must be positive",
)
@icontract.require(
    lambda mutual_incoherence: mutual_incoherence >= 0.0,
    "mutual incoherence must be non-negative",
)
@icontract.ensure(
    lambda result: result["confidence"] in {"high", "medium", "absolute"},
    "confidence must be high, medium, or absolute",
)
@icontract.ensure(
    lambda result: len(result["source_sections"]) > 0,
    "must cite at least one source section",
)
def recommend_regularization(
    condition_number: float,
    n_p_ratio: float,
    mutual_incoherence: float,
    lasso_viable: bool,
) -> dict:
    """Recommend a regularization strategy based on design matrix diagnostics.

    Encodes the decision table from heuristics.pdf Section A. The condition
    number and n/p ratio determine whether OLS is viable; mutual incoherence
    and the Lasso sample complexity bound determine L1 vs L2 vs ElasticNet.

    Thresholds: kappa(X) > 30 -> regularize, mu <= 1 -> Lasso viable
    Source: Hoerl & Kennard (1970), Wainwright (2009); heuristics.pdf Section A

    Args:
        condition_number: kappa(X) from compute_condition_number.
        n_p_ratio: n/p from compute_n_p_ratio.
        mutual_incoherence: mu from compute_mutual_incoherence.
        lasso_viable: from check_lasso_sample_complexity.

    Returns:
        Recommendation dict with regularization strategy.
    """
    thresholds = {
        "OLS_INSTABILITY_LIMIT": OLS_INSTABILITY_LIMIT,
        "LASSO_MUTUAL_INCOHERENCE_LIMIT": LASSO_MUTUAL_INCOHERENCE_LIMIT,
        "N_P_RATIO_OLS_VIABLE": N_P_RATIO_OLS_VIABLE,
    }

    if (
        condition_number < OLS_INSTABILITY_LIMIT
        and n_p_ratio >= N_P_RATIO_OLS_VIABLE
    ):
        return {
            "recommendation": "OLS (no regularization)",
            "confidence": "high",
            "sklearn_class": "sklearn.linear_model.LinearRegression",
            "reasoning": (
                f"Condition number {condition_number:.1f} < {OLS_INSTABILITY_LIMIT} "
                f"and n/p ratio {n_p_ratio:.1f} >= {N_P_RATIO_OLS_VIABLE}, "
                "so OLS is stable."
            ),
            "alternatives": [
                "sklearn.linear_model.Ridge",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["A"],
        }

    if mutual_incoherence > LASSO_MUTUAL_INCOHERENCE_LIMIT:
        return {
            "recommendation": "ElasticNet or Ridge",
            "confidence": "high",
            "sklearn_class": "sklearn.linear_model.ElasticNet",
            "reasoning": (
                f"Mutual incoherence {mutual_incoherence:.2f} > "
                f"{LASSO_MUTUAL_INCOHERENCE_LIMIT} violates the irrepresentable "
                "condition; Lasso variable selection is unreliable."
            ),
            "alternatives": [
                "sklearn.linear_model.Ridge",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["A"],
        }

    if (
        lasso_viable
        and mutual_incoherence <= LASSO_MUTUAL_INCOHERENCE_LIMIT
    ):
        return {
            "recommendation": "Lasso (L1)",
            "confidence": "medium",
            "sklearn_class": "sklearn.linear_model.Lasso",
            "reasoning": (
                f"Mutual incoherence {mutual_incoherence:.2f} <= "
                f"{LASSO_MUTUAL_INCOHERENCE_LIMIT} and sample complexity "
                "bound satisfied; Lasso recovery is viable."
            ),
            "alternatives": [
                "sklearn.linear_model.ElasticNet",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["A"],
        }

    return {
        "recommendation": "Ridge (L2)",
        "confidence": "high",
        "sklearn_class": "sklearn.linear_model.Ridge",
        "reasoning": (
            f"Condition number {condition_number:.1f} >= {OLS_INSTABILITY_LIMIT} "
            "and Lasso conditions not met; Ridge stabilizes via diagonal inflation."
        ),
        "alternatives": [
            "sklearn.linear_model.ElasticNet",
        ],
        "thresholds_applied": thresholds,
        "source_sections": ["A"],
    }


# ---------------------------------------------------------------------------
# Section B: Loss function recommendation
# ---------------------------------------------------------------------------


@register_atom(witness_recommend_loss_function)
@icontract.require(
    lambda excess_kurtosis: np.isfinite(excess_kurtosis),
    "excess kurtosis must be finite",
)
@icontract.require(
    lambda residual_kurtosis: np.isfinite(residual_kurtosis),
    "residual kurtosis must be finite",
)
@icontract.ensure(
    lambda result: result["confidence"] in {"high", "medium", "absolute"},
    "confidence must be high, medium, or absolute",
)
def recommend_loss_function(
    excess_kurtosis: float,
    residual_kurtosis: float,
) -> dict:
    """Recommend a loss function based on tail heaviness of target and residuals.

    Encodes the decision table from heuristics.pdf Section B. Excess
    kurtosis > 1.0 in target or residuals indicates heavy tails requiring
    robust loss (Huber or MAE) instead of squared error.

    Threshold: excess kurtosis > 1.0 for robust loss
    Source: heuristics.pdf Section B

    Args:
        excess_kurtosis: From compute_excess_kurtosis on y.
        residual_kurtosis: From compute_residual_kurtosis on OLS residuals.

    Returns:
        Recommendation dict with loss function.
    """
    thresholds = {
        "KURTOSIS_ROBUST_LOSS_THRESHOLD": KURTOSIS_ROBUST_LOSS_THRESHOLD,
    }

    if residual_kurtosis <= KURTOSIS_ROBUST_LOSS_THRESHOLD:
        return {
            "recommendation": "Squared Error",
            "confidence": "high",
            "sklearn_class": "squared_error",
            "reasoning": (
                f"Residual kurtosis {residual_kurtosis:.2f} <= "
                f"{KURTOSIS_ROBUST_LOSS_THRESHOLD}; residuals are "
                "near-Gaussian, squared error is appropriate."
            ),
            "alternatives": [],
            "thresholds_applied": thresholds,
            "source_sections": ["B"],
        }

    if excess_kurtosis > KURTOSIS_ROBUST_LOSS_THRESHOLD * 3:
        return {
            "recommendation": "Absolute Error (MAE)",
            "confidence": "high",
            "sklearn_class": "absolute_error",
            "reasoning": (
                f"Target kurtosis {excess_kurtosis:.2f} is extremely heavy-tailed; "
                "MAE is fully robust to outliers."
            ),
            "alternatives": [
                "huber",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["B"],
        }

    return {
        "recommendation": "Huber Loss",
        "confidence": "high",
        "sklearn_class": "huber",
        "reasoning": (
            f"Residual kurtosis {residual_kurtosis:.2f} > "
            f"{KURTOSIS_ROBUST_LOSS_THRESHOLD}; tails are heavy "
            "but Huber bridges MSE precision with MAE robustness."
        ),
        "alternatives": [
            "absolute_error",
            "squared_error",
        ],
        "thresholds_applied": thresholds,
        "source_sections": ["B"],
    }


# ---------------------------------------------------------------------------
# Section C: Linear model recommendation
# ---------------------------------------------------------------------------


@register_atom(witness_recommend_linear_model)
@icontract.require(lambda n: n >= 1, "n must be at least 1")
@icontract.require(lambda p: p >= 1, "p must be at least 1")
@icontract.ensure(
    lambda result: result["confidence"] in {"high", "medium", "absolute"},
    "confidence must be high, medium, or absolute",
)
def recommend_linear_model(
    n: int,
    p: int,
    is_sparse: bool,
    dispersion_index: float,
    tweedie_power: float,
) -> dict:
    """Recommend a linear model class and solver based on data characteristics.

    Encodes the decision table from heuristics.pdf Section C. Selects
    between Gaussian, Poisson, Gamma, and Tweedie regressors based on
    the dispersion index and estimated Tweedie power, and chooses the
    solver based on matrix dimensions and sparsity.

    Thresholds: DI ~= 1 for Poisson, DI > 1.1 for overdispersion,
    p > 1000 for solver selection
    Source: heuristics.pdf Section C

    Args:
        n: Number of samples.
        p: Number of features.
        is_sparse: Whether the design matrix is sparse.
        dispersion_index: DI = Var(y) / E(y) from compute_dispersion_index.
        tweedie_power: Estimated power parameter from estimate_tweedie_power.

    Returns:
        Recommendation dict with linear model class and solver.
    """
    thresholds = {
        "DISPERSION_INDEX_POISSON": DISPERSION_INDEX_POISSON,
        "DISPERSION_OVERDISPERSION": DISPERSION_OVERDISPERSION,
        "SOLVER_FEATURE_LIMIT": SOLVER_FEATURE_LIMIT,
    }

    # Determine solver
    if is_sparse:
        solver = "saga"
        solver_reason = "sparse matrix requires saga solver"
    elif p > SOLVER_FEATURE_LIMIT:
        solver = "sag"
        solver_reason = f"p={p} > {SOLVER_FEATURE_LIMIT}; sag avoids full Hessian"
    else:
        solver = "newton-cholesky"
        solver_reason = f"p={p} <= {SOLVER_FEATURE_LIMIT}; newton-cholesky is exact"

    # Determine model family
    if dispersion_index > DISPERSION_OVERDISPERSION and 1.0 < tweedie_power < 2.0:
        return {
            "recommendation": f"TweedieRegressor (power={tweedie_power:.2f})",
            "confidence": "high",
            "sklearn_class": "sklearn.linear_model.TweedieRegressor",
            "reasoning": (
                f"DI={dispersion_index:.2f} > {DISPERSION_OVERDISPERSION} with "
                f"Tweedie power {tweedie_power:.2f} in (1,2); compound Poisson-Gamma. "
                f"Solver: {solver} ({solver_reason})."
            ),
            "alternatives": [
                "sklearn.linear_model.GammaRegressor",
                "sklearn.linear_model.PoissonRegressor",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["C"],
        }

    if tweedie_power >= 2.0:
        return {
            "recommendation": "GammaRegressor",
            "confidence": "high",
            "sklearn_class": "sklearn.linear_model.GammaRegressor",
            "reasoning": (
                f"Tweedie power {tweedie_power:.2f} >= 2.0; Var ~ Mean^2 "
                f"indicates Gamma family. Solver: {solver} ({solver_reason})."
            ),
            "alternatives": [
                "sklearn.linear_model.TweedieRegressor",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["C"],
        }

    if (
        abs(dispersion_index - DISPERSION_INDEX_POISSON) < DISPERSION_OVERDISPERSION - DISPERSION_INDEX_POISSON
    ):
        return {
            "recommendation": "PoissonRegressor",
            "confidence": "high",
            "sklearn_class": "sklearn.linear_model.PoissonRegressor",
            "reasoning": (
                f"DI={dispersion_index:.2f} ~ {DISPERSION_INDEX_POISSON}; "
                f"variance ~ mean consistent with Poisson. "
                f"Solver: {solver} ({solver_reason})."
            ),
            "alternatives": [
                "sklearn.linear_model.TweedieRegressor",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["C"],
        }

    return {
        "recommendation": f"LinearRegression (solver={solver})",
        "confidence": "high",
        "sklearn_class": "sklearn.linear_model.LinearRegression",
        "reasoning": (
            f"No count/positive-only signal detected (DI={dispersion_index:.2f}). "
            f"Default to Gaussian linear model. Solver: {solver} ({solver_reason})."
        ),
        "alternatives": [
            "sklearn.linear_model.Ridge",
        ],
        "thresholds_applied": thresholds,
        "source_sections": ["C"],
    }


# ---------------------------------------------------------------------------
# Section D: Tree ensemble recommendation
# ---------------------------------------------------------------------------


@register_atom(witness_recommend_tree_ensemble)
@icontract.require(lambda n: n >= 1, "n must be at least 1")
@icontract.require(lambda n_categorical: n_categorical >= 0, "categorical count must be non-negative")
@icontract.require(
    lambda noise_level: np.isfinite(noise_level) and noise_level >= 0.0,
    "noise level must be non-negative and finite",
)
@icontract.ensure(
    lambda result: result["confidence"] in {"high", "medium", "absolute"},
    "confidence must be high, medium, or absolute",
)
def recommend_tree_ensemble(
    n: int,
    n_categorical: int,
    noise_level: float,
) -> dict:
    """Recommend a tree ensemble type based on sample size, categoricals, and noise.

    Encodes the decision table from heuristics.pdf Section D. Large datasets
    or datasets with categoricals favor HistGradientBoosting. High noise
    favors RandomForest (bagging reduces variance). Low noise favors
    GradientBoosting (boosting reduces bias).

    Threshold: n > 10,000 for HistGradientBoosting
    Source: Friedman (1999); heuristics.pdf Section D

    Args:
        n: Number of samples.
        n_categorical: Number of categorical features.
        noise_level: Estimated noise variance from estimate_noise_level.

    Returns:
        Recommendation dict with tree ensemble type.
    """
    thresholds = {
        "HIST_GB_SAMPLE_BOUNDARY": HIST_GB_SAMPLE_BOUNDARY,
    }

    if n > HIST_GB_SAMPLE_BOUNDARY:
        return {
            "recommendation": "HistGradientBoosting",
            "confidence": "high",
            "sklearn_class": "sklearn.ensemble.HistGradientBoostingRegressor",
            "reasoning": (
                f"n={n} > {HIST_GB_SAMPLE_BOUNDARY}; histogram binning is "
                "O(n*bins) vs O(n*p*log(n)) for exact splits."
            ),
            "alternatives": [
                "sklearn.ensemble.GradientBoostingRegressor",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["D"],
        }

    if n_categorical > 0:
        return {
            "recommendation": "HistGradientBoosting (native categoricals)",
            "confidence": "high",
            "sklearn_class": "sklearn.ensemble.HistGradientBoostingRegressor",
            "reasoning": (
                f"{n_categorical} categorical features detected; "
                "HistGradientBoosting handles them natively without encoding."
            ),
            "alternatives": [
                "sklearn.ensemble.RandomForestRegressor",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["D"],
        }

    # Use noise level relative to target variance as a rough indicator.
    # High noise -> bagging (RandomForest), low noise -> boosting.
    # We use a simple heuristic: if noise_level > median threshold,
    # prefer RandomForest for variance reduction.
    if noise_level > 0.5:
        return {
            "recommendation": "RandomForest (bagging)",
            "confidence": "medium",
            "sklearn_class": "sklearn.ensemble.RandomForestRegressor",
            "reasoning": (
                f"Noise level {noise_level:.3f} is elevated; bagging reduces "
                "variance by averaging independent deep trees."
            ),
            "alternatives": [
                "sklearn.ensemble.ExtraTreesRegressor",
                "sklearn.ensemble.GradientBoostingRegressor",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["D"],
        }

    return {
        "recommendation": "GradientBoosting",
        "confidence": "high",
        "sklearn_class": "sklearn.ensemble.GradientBoostingRegressor",
        "reasoning": (
            f"Noise level {noise_level:.3f} is low; boosting targets bias "
            "reduction via sequential residual fitting."
        ),
        "alternatives": [
            "sklearn.ensemble.HistGradientBoostingRegressor",
        ],
        "thresholds_applied": thresholds,
        "source_sections": ["D"],
    }


# ---------------------------------------------------------------------------
# Section E: Preprocessing recommendation
# ---------------------------------------------------------------------------


@register_atom(witness_recommend_preprocessing)
@icontract.require(
    lambda skewness_array: len(skewness_array) >= 1,
    "skewness array must have at least one element",
)
@icontract.require(
    lambda vif_array: len(vif_array) >= 1,
    "VIF array must have at least one element",
)
@icontract.ensure(
    lambda result: result["confidence"] in {"high", "medium", "absolute"},
    "confidence must be high, medium, or absolute",
)
def recommend_preprocessing(
    skewness_array: NDArray[np.float64],
    vif_array: NDArray[np.float64],
    is_sparse: bool,
    model_requires_scaling: bool,
) -> dict:
    """Recommend preprocessing transformations based on feature diagnostics.

    Encodes the decision table from heuristics.pdf Section E. Highly
    skewed features need PowerTransformer, high VIF needs PCA or feature
    removal, sparse matrices must avoid mean centering.

    Thresholds: |skewness| > 1 for transform, VIF > 5 moderate, > 10 severe
    Source: heuristics.pdf Section E

    Args:
        skewness_array: Per-feature skewness values.
        vif_array: Per-feature VIF values.
        is_sparse: Whether the design matrix is sparse.
        model_requires_scaling: Whether the downstream model needs scaled features.

    Returns:
        Recommendation dict with preprocessing steps.
    """
    thresholds = {
        "SKEWNESS_TRANSFORM_THRESHOLD": SKEWNESS_TRANSFORM_THRESHOLD,
        "VIF_MODERATE_COLLINEARITY": VIF_MODERATE_COLLINEARITY,
        "VIF_SEVERE_COLLINEARITY": VIF_SEVERE_COLLINEARITY,
    }

    steps: list[str] = []
    sklearn_classes: list[str] = []
    reasons: list[str] = []

    # Skewness check
    n_skewed = int(np.sum(np.abs(skewness_array) > SKEWNESS_TRANSFORM_THRESHOLD))
    if n_skewed > 0:
        steps.append("PowerTransformer")
        sklearn_classes.append("sklearn.preprocessing.PowerTransformer")
        reasons.append(
            f"{n_skewed}/{len(skewness_array)} features have |skewness| > "
            f"{SKEWNESS_TRANSFORM_THRESHOLD}"
        )

    # VIF check
    max_vif = float(np.max(vif_array))
    n_severe = int(np.sum(vif_array > VIF_SEVERE_COLLINEARITY))
    n_moderate = int(np.sum(vif_array > VIF_MODERATE_COLLINEARITY))
    if n_severe > 0:
        steps.append("PCA or feature removal")
        sklearn_classes.append("sklearn.decomposition.PCA")
        reasons.append(
            f"{n_severe} features have VIF > {VIF_SEVERE_COLLINEARITY} "
            f"(max VIF={max_vif:.1f}); severe collinearity"
        )
    elif n_moderate > 0:
        steps.append("Ridge or moderate VIF monitoring")
        reasons.append(
            f"{n_moderate} features have VIF > {VIF_MODERATE_COLLINEARITY} "
            f"(max VIF={max_vif:.1f}); moderate collinearity"
        )

    # Scaling check
    if model_requires_scaling:
        if is_sparse:
            steps.append("MaxAbsScaler (sparse-safe)")
            sklearn_classes.append("sklearn.preprocessing.MaxAbsScaler")
            reasons.append("Sparse matrix: avoid mean centering, use MaxAbsScaler")
        else:
            steps.append("StandardScaler")
            sklearn_classes.append("sklearn.preprocessing.StandardScaler")
            reasons.append("Dense matrix with scaling-dependent model")

    if not steps:
        steps.append("No preprocessing required")
        reasons.append("No skewness, collinearity, or scaling issues detected")

    return {
        "recommendation": " + ".join(steps),
        "confidence": "high",
        "sklearn_class": sklearn_classes[0] if sklearn_classes else "none",
        "reasoning": "; ".join(reasons) + ".",
        "alternatives": [],
        "thresholds_applied": thresholds,
        "source_sections": ["E"],
    }


# ---------------------------------------------------------------------------
# Section F: Dimensionality reduction recommendation
# ---------------------------------------------------------------------------


@register_atom(witness_recommend_dimensionality_reduction)
@icontract.require(
    lambda condition_number: condition_number >= 1.0,
    "condition number must be >= 1",
)
@icontract.require(
    lambda explained_variance: 0.0 <= explained_variance <= 1.0,
    "explained variance must be in [0, 1]",
)
@icontract.ensure(
    lambda result: result["confidence"] in {"high", "medium", "absolute"},
    "confidence must be high, medium, or absolute",
)
def recommend_dimensionality_reduction(
    condition_number: float,
    is_sparse: bool,
    explained_variance: float,
) -> dict:
    """Recommend dimensionality reduction based on condition number and sparsity.

    Encodes the decision table from heuristics.pdf Section F. High condition
    number or low explained variance ratio signals the need for PCA
    (dense) or TruncatedSVD (sparse).

    Threshold: condition number > 30, explained variance default 0.95
    Source: heuristics.pdf Section F

    Args:
        condition_number: kappa(X) from compute_condition_number.
        is_sparse: Whether the design matrix is sparse.
        explained_variance: Fraction explained by top components.

    Returns:
        Recommendation dict with dimensionality reduction strategy.
    """
    thresholds = {
        "OLS_INSTABILITY_LIMIT": OLS_INSTABILITY_LIMIT,
        "EXPLAINED_VARIANCE_DEFAULT": EXPLAINED_VARIANCE_DEFAULT,
    }

    needs_reduction = (
        condition_number >= OLS_INSTABILITY_LIMIT
        or explained_variance < EXPLAINED_VARIANCE_DEFAULT
    )

    if not needs_reduction:
        return {
            "recommendation": "No dimensionality reduction needed",
            "confidence": "high",
            "sklearn_class": "none",
            "reasoning": (
                f"Condition number {condition_number:.1f} < {OLS_INSTABILITY_LIMIT} "
                f"and explained variance {explained_variance:.2f} >= "
                f"{EXPLAINED_VARIANCE_DEFAULT}; full feature space is viable."
            ),
            "alternatives": [],
            "thresholds_applied": thresholds,
            "source_sections": ["F"],
        }

    if is_sparse:
        return {
            "recommendation": "TruncatedSVD",
            "confidence": "absolute",
            "sklearn_class": "sklearn.decomposition.TruncatedSVD",
            "reasoning": (
                "Sparse matrix requires TruncatedSVD; PCA mean-centering "
                "would destroy sparsity and exhaust memory."
            ),
            "alternatives": [],
            "thresholds_applied": thresholds,
            "source_sections": ["F"],
        }

    return {
        "recommendation": "PCA",
        "confidence": "high",
        "sklearn_class": "sklearn.decomposition.PCA",
        "reasoning": (
            f"Condition number {condition_number:.1f} >= {OLS_INSTABILITY_LIMIT} "
            "or low explained variance; PCA produces orthogonal components "
            "with condition number = 1."
        ),
        "alternatives": [
            "sklearn.decomposition.TruncatedSVD",
        ],
        "thresholds_applied": thresholds,
        "source_sections": ["F"],
    }


# ---------------------------------------------------------------------------
# Section G: Hyperparameter range recommendation
# ---------------------------------------------------------------------------


@register_atom(witness_recommend_hyperparameter_ranges)
@icontract.require(
    lambda model_type: model_type in {
        "lasso", "elasticnet", "ridge",
        "gradient_boosting", "random_forest", "hist_gradient_boosting",
    },
    "model_type must be a recognized sklearn model family",
)
@icontract.require(lambda n: n >= 1, "n must be at least 1")
@icontract.require(lambda p: p >= 1, "p must be at least 1")
@icontract.ensure(
    lambda result: result["confidence"] in {"high", "medium", "absolute"},
    "confidence must be high, medium, or absolute",
)
def recommend_hyperparameter_ranges(
    model_type: str,
    n: int,
    p: int,
) -> dict:
    """Recommend hyperparameter search ranges for a given model type.

    Encodes the dynamic range heuristics from heuristics.pdf Section G.
    Alpha ranges for regularized models use log-space grids with a hard
    floor. Boosting ranges inversely scale learning rate with estimators.

    Thresholds: alpha floor = 1e-5, lr < 0.125 diminishing returns
    Source: Friedman (1999); heuristics.pdf Section G

    Args:
        model_type: One of "lasso", "elasticnet", "ridge",
            "gradient_boosting", "random_forest", "hist_gradient_boosting".
        n: Number of samples.
        p: Number of features.

    Returns:
        Recommendation dict with suggested hyperparameter ranges.
    """
    thresholds = {
        "ALPHA_FLOOR": ALPHA_FLOOR,
        "BOOSTING_DIMINISHING_LR": BOOSTING_DIMINISHING_LR,
        "BOOSTING_FLATLINE_ESTIMATORS": BOOSTING_FLATLINE_ESTIMATORS,
    }

    if model_type in {"lasso", "elasticnet", "ridge"}:
        return {
            "recommendation": f"Log-space alpha grid for {model_type}",
            "confidence": "high",
            "sklearn_class": f"sklearn.linear_model.{model_type.title().replace('_', '')}",
            "reasoning": (
                f"sklearn's cost function scales residuals by 1/(2n); alpha is "
                f"sample-invariant. Log-space [{ALPHA_FLOOR}, 1.0] with 50 points."
            ),
            "alternatives": [],
            "thresholds_applied": thresholds,
            "source_sections": ["G"],
            "ranges": {
                "alpha": {"low": ALPHA_FLOOR, "high": 1.0, "scale": "log", "n_points": 50},
            },
        }

    if model_type == "gradient_boosting":
        max_depth = max(2, int(np.log2(p + 1)))
        return {
            "recommendation": "Boosting grid with inverse lr/n_estimators tradeoff",
            "confidence": "high",
            "sklearn_class": "sklearn.ensemble.GradientBoostingRegressor",
            "reasoning": (
                f"lr in [{BOOSTING_DIMINISHING_LR}, 0.3] with n_estimators "
                f"up to {BOOSTING_FLATLINE_ESTIMATORS}; max_depth ~ log2(p)={max_depth}."
            ),
            "alternatives": [],
            "thresholds_applied": thresholds,
            "source_sections": ["G"],
            "ranges": {
                "learning_rate": {"low": BOOSTING_DIMINISHING_LR, "high": 0.3, "scale": "log"},
                "n_estimators": {"low": 50, "high": BOOSTING_FLATLINE_ESTIMATORS, "scale": "linear"},
                "max_depth": {"low": 2, "high": max_depth, "scale": "linear"},
            },
        }

    if model_type == "random_forest":
        max_depth = max(3, int(np.log2(p + 1)) + 2)
        return {
            "recommendation": "RandomForest depth and estimator grid",
            "confidence": "high",
            "sklearn_class": "sklearn.ensemble.RandomForestRegressor",
            "reasoning": (
                f"max_depth up to log2(p)+2={max_depth}; n_estimators in [100, 500]."
            ),
            "alternatives": [],
            "thresholds_applied": thresholds,
            "source_sections": ["G"],
            "ranges": {
                "n_estimators": {"low": 100, "high": 500, "scale": "linear"},
                "max_depth": {"low": 3, "high": max_depth, "scale": "linear"},
            },
        }

    # hist_gradient_boosting
    max_depth = max(3, int(np.log2(p + 1)))
    return {
        "recommendation": "HistGradientBoosting grid",
        "confidence": "high",
        "sklearn_class": "sklearn.ensemble.HistGradientBoostingRegressor",
        "reasoning": (
            f"lr in [{BOOSTING_DIMINISHING_LR}, 0.3] with max_iter "
            f"up to {BOOSTING_FLATLINE_ESTIMATORS}; max_depth ~ log2(p)={max_depth}."
        ),
        "alternatives": [],
        "thresholds_applied": thresholds,
        "source_sections": ["G"],
        "ranges": {
            "learning_rate": {"low": BOOSTING_DIMINISHING_LR, "high": 0.3, "scale": "log"},
            "max_iter": {"low": 50, "high": BOOSTING_FLATLINE_ESTIMATORS, "scale": "linear"},
            "max_depth": {"low": 3, "high": max_depth, "scale": "linear"},
        },
    }


# ---------------------------------------------------------------------------
# Section H: Cross-validation strategy recommendation
# ---------------------------------------------------------------------------


@register_atom(witness_recommend_cv_strategy)
@icontract.require(lambda n: n >= 1, "n must be at least 1")
@icontract.ensure(
    lambda result: result["confidence"] in {"high", "medium", "absolute"},
    "confidence must be high, medium, or absolute",
)
def recommend_cv_strategy(
    n: int,
    is_classification: bool,
    is_timeseries: bool,
    has_groups: bool,
) -> dict:
    """Recommend a cross-validation splitting strategy.

    Encodes the decision table from heuristics.pdf Section H. Time series
    data requires TimeSeriesSplit. Classification with imbalance requires
    StratifiedKFold. Grouped data requires GroupKFold. Small datasets
    may use LOOCV.

    Threshold: n <= 100 for LOOCV viability
    Source: Varma & Simon (2006); heuristics.pdf Section H

    Args:
        n: Number of samples.
        is_classification: Whether the task is classification.
        is_timeseries: Whether the data has temporal ordering.
        has_groups: Whether samples belong to identifiable groups.

    Returns:
        Recommendation dict with CV strategy.
    """
    thresholds = {
        "LOOCV_VIABILITY_BOUNDARY": LOOCV_VIABILITY_BOUNDARY,
    }

    if is_timeseries:
        return {
            "recommendation": "TimeSeriesSplit",
            "confidence": "absolute",
            "sklearn_class": "sklearn.model_selection.TimeSeriesSplit",
            "reasoning": (
                "Temporal data detected; standard KFold would leak future "
                "information into training folds."
            ),
            "alternatives": [],
            "thresholds_applied": thresholds,
            "source_sections": ["H"],
        }

    if has_groups:
        return {
            "recommendation": "GroupKFold",
            "confidence": "high",
            "sklearn_class": "sklearn.model_selection.GroupKFold",
            "reasoning": (
                "Grouped data detected; GroupKFold prevents same-group "
                "contamination between train and test folds."
            ),
            "alternatives": [
                "sklearn.model_selection.LeaveOneGroupOut",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["H"],
        }

    if n <= LOOCV_VIABILITY_BOUNDARY:
        return {
            "recommendation": "LeaveOneOut (LOOCV)",
            "confidence": "medium",
            "sklearn_class": "sklearn.model_selection.LeaveOneOut",
            "reasoning": (
                f"n={n} <= {LOOCV_VIABILITY_BOUNDARY}; LOOCV maximizes data "
                "utilization with nearly unbiased error estimates."
            ),
            "alternatives": [
                "sklearn.model_selection.RepeatedKFold",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["H"],
        }

    if is_classification:
        return {
            "recommendation": "StratifiedKFold",
            "confidence": "high",
            "sklearn_class": "sklearn.model_selection.StratifiedKFold",
            "reasoning": (
                "Classification task with sufficient samples; StratifiedKFold "
                "preserves class proportions across folds."
            ),
            "alternatives": [
                "sklearn.model_selection.RepeatedStratifiedKFold",
            ],
            "thresholds_applied": thresholds,
            "source_sections": ["H"],
        }

    return {
        "recommendation": "KFold (k=5)",
        "confidence": "high",
        "sklearn_class": "sklearn.model_selection.KFold",
        "reasoning": (
            f"Regression with n={n} > {LOOCV_VIABILITY_BOUNDARY}; "
            "standard 5-fold CV provides stable error estimates."
        ),
        "alternatives": [
            "sklearn.model_selection.RepeatedKFold",
        ],
        "thresholds_applied": thresholds,
        "source_sections": ["H"],
    }
