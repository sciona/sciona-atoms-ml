"""Ghost witnesses for model selection recommendation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_recommend_regularization(
    condition_number: float,
    n_p_ratio: float,
    mutual_incoherence: float,
    lasso_viable: bool,
) -> dict:
    """Regularization recommendation is a dict with strategy metadata."""
    return {
        "recommendation": "Ridge (L2)",
        "confidence": "high",
        "sklearn_class": "sklearn.linear_model.Ridge",
        "reasoning": "Witness placeholder.",
        "alternatives": [],
        "thresholds_applied": {},
        "source_sections": ["A"],
    }


def witness_recommend_loss_function(
    excess_kurtosis: float,
    residual_kurtosis: float,
) -> dict:
    """Loss function recommendation is a dict with loss metadata."""
    return {
        "recommendation": "Squared Error",
        "confidence": "high",
        "sklearn_class": "squared_error",
        "reasoning": "Witness placeholder.",
        "alternatives": [],
        "thresholds_applied": {},
        "source_sections": ["B"],
    }


def witness_recommend_linear_model(
    n: int,
    p: int,
    is_sparse: bool,
    dispersion_index: float,
    tweedie_power: float,
) -> dict:
    """Linear model recommendation is a dict with model and solver metadata."""
    return {
        "recommendation": "LinearRegression",
        "confidence": "high",
        "sklearn_class": "sklearn.linear_model.LinearRegression",
        "reasoning": "Witness placeholder.",
        "alternatives": [],
        "thresholds_applied": {},
        "source_sections": ["C"],
    }


def witness_recommend_tree_ensemble(
    n: int,
    n_categorical: int,
    noise_level: float,
) -> dict:
    """Tree ensemble recommendation is a dict with ensemble metadata."""
    return {
        "recommendation": "GradientBoosting",
        "confidence": "high",
        "sklearn_class": "sklearn.ensemble.GradientBoostingRegressor",
        "reasoning": "Witness placeholder.",
        "alternatives": [],
        "thresholds_applied": {},
        "source_sections": ["D"],
    }


def witness_recommend_preprocessing(
    skewness_array: AbstractArray,
    vif_array: AbstractArray,
    is_sparse: bool,
    model_requires_scaling: bool,
) -> dict:
    """Preprocessing recommendation is a dict with transform pipeline metadata."""
    return {
        "recommendation": "StandardScaler",
        "confidence": "high",
        "sklearn_class": "sklearn.preprocessing.StandardScaler",
        "reasoning": "Witness placeholder.",
        "alternatives": [],
        "thresholds_applied": {},
        "source_sections": ["E"],
    }


def witness_recommend_dimensionality_reduction(
    condition_number: float,
    is_sparse: bool,
    explained_variance: float,
) -> dict:
    """Dimensionality reduction recommendation is a dict with strategy metadata."""
    return {
        "recommendation": "PCA",
        "confidence": "high",
        "sklearn_class": "sklearn.decomposition.PCA",
        "reasoning": "Witness placeholder.",
        "alternatives": [],
        "thresholds_applied": {},
        "source_sections": ["F"],
    }


def witness_recommend_hyperparameter_ranges(
    model_type: str,
    n: int,
    p: int,
) -> dict:
    """Hyperparameter range recommendation is a dict with range metadata."""
    return {
        "recommendation": "Log-space alpha grid",
        "confidence": "high",
        "sklearn_class": "sklearn.linear_model.Lasso",
        "reasoning": "Witness placeholder.",
        "alternatives": [],
        "thresholds_applied": {},
        "source_sections": ["G"],
    }


def witness_recommend_cv_strategy(
    n: int,
    is_classification: bool,
    is_timeseries: bool,
    has_groups: bool,
) -> dict:
    """CV strategy recommendation is a dict with splitting metadata."""
    return {
        "recommendation": "KFold",
        "confidence": "high",
        "sklearn_class": "sklearn.model_selection.KFold",
        "reasoning": "Witness placeholder.",
        "alternatives": [],
        "thresholds_applied": {},
        "source_sections": ["H"],
    }
