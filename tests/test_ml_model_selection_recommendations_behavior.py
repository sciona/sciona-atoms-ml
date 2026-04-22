"""Behavioral tests for model selection recommendation atoms.

Verifies that deterministic decision logic matches expected outcomes
for canonical scenarios from heuristics.pdf.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_recommendation_atoms_import() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import (
        recommend_regularization,
        recommend_loss_function,
        recommend_linear_model,
        recommend_tree_ensemble,
        recommend_preprocessing,
        recommend_dimensionality_reduction,
        recommend_hyperparameter_ranges,
        recommend_cv_strategy,
    )
    assert callable(recommend_regularization)
    assert callable(recommend_cv_strategy)


# --- Section A: Regularization ---


def test_ols_recommended_when_well_conditioned_and_high_np_ratio() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_regularization

    r = recommend_regularization(
        condition_number=5.0, n_p_ratio=20.0, mutual_incoherence=0.3, lasso_viable=True
    )
    assert "OLS" in r["recommendation"]
    assert r["confidence"] == "high"
    assert "LinearRegression" in r["sklearn_class"]


def test_ridge_recommended_for_high_condition_number() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_regularization

    r = recommend_regularization(
        condition_number=100.0, n_p_ratio=5.0, mutual_incoherence=0.5, lasso_viable=False
    )
    assert "Ridge" in r["recommendation"]


def test_lasso_recommended_when_viable_and_incoherent() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_regularization

    r = recommend_regularization(
        condition_number=50.0, n_p_ratio=5.0, mutual_incoherence=0.5, lasso_viable=True
    )
    assert "Lasso" in r["recommendation"]


def test_elasticnet_recommended_when_incoherence_violated() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_regularization

    r = recommend_regularization(
        condition_number=50.0, n_p_ratio=5.0, mutual_incoherence=1.5, lasso_viable=True
    )
    assert "ElasticNet" in r["recommendation"] or "Ridge" in r["recommendation"]


# --- Section B: Loss function ---


def test_squared_error_for_gaussian_residuals() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_loss_function

    r = recommend_loss_function(excess_kurtosis=0.3, residual_kurtosis=0.5)
    assert "Squared" in r["recommendation"]
    assert r["sklearn_class"] == "squared_error"


def test_huber_for_moderately_heavy_tails() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_loss_function

    r = recommend_loss_function(excess_kurtosis=2.0, residual_kurtosis=2.0)
    assert "Huber" in r["recommendation"]
    assert r["sklearn_class"] == "huber"


def test_absolute_error_for_extremely_heavy_tails() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_loss_function

    r = recommend_loss_function(excess_kurtosis=5.0, residual_kurtosis=5.0)
    assert "Absolute" in r["recommendation"] or "MAE" in r["recommendation"]


# --- Section C: Linear model ---


def test_poisson_for_count_data() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_linear_model

    r = recommend_linear_model(n=1000, p=10, is_sparse=False, dispersion_index=1.0, tweedie_power=1.0)
    assert "Poisson" in r["recommendation"]


def test_gamma_for_high_tweedie_power() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_linear_model

    r = recommend_linear_model(n=1000, p=10, is_sparse=False, dispersion_index=2.0, tweedie_power=2.0)
    assert "Gamma" in r["recommendation"]


def test_tweedie_for_compound_poisson_gamma() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_linear_model

    r = recommend_linear_model(n=1000, p=10, is_sparse=False, dispersion_index=1.5, tweedie_power=1.5)
    assert "Tweedie" in r["recommendation"]


# --- Section D: Tree ensemble ---


def test_histgb_for_large_datasets() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_tree_ensemble

    r = recommend_tree_ensemble(n=50000, n_categorical=0, noise_level=0.1)
    assert "Hist" in r["recommendation"]
    assert r["confidence"] == "high"


def test_random_forest_for_noisy_data() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_tree_ensemble

    r = recommend_tree_ensemble(n=500, n_categorical=0, noise_level=1.0)
    assert "Random" in r["recommendation"] or "Forest" in r["recommendation"]


def test_gradient_boosting_for_clean_data() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_tree_ensemble

    r = recommend_tree_ensemble(n=500, n_categorical=0, noise_level=0.1)
    assert "GradientBoosting" in r["recommendation"]


# --- Section E: Preprocessing ---


def test_preprocessing_skewed_features() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_preprocessing

    r = recommend_preprocessing(
        skewness_array=np.array([2.5, 0.1, -3.0]),
        vif_array=np.array([1.5, 2.0, 1.2]),
        is_sparse=False,
        model_requires_scaling=False,
    )
    assert "PowerTransformer" in r["recommendation"]


def test_preprocessing_severe_collinearity() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_preprocessing

    r = recommend_preprocessing(
        skewness_array=np.array([0.1, 0.2]),
        vif_array=np.array([15.0, 12.0]),
        is_sparse=False,
        model_requires_scaling=False,
    )
    assert "PCA" in r["recommendation"] or "feature removal" in r["recommendation"]


def test_preprocessing_sparse_scaling() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_preprocessing

    r = recommend_preprocessing(
        skewness_array=np.array([0.1]),
        vif_array=np.array([1.5]),
        is_sparse=True,
        model_requires_scaling=True,
    )
    assert "MaxAbsScaler" in r["recommendation"]


# --- Section F: Dimensionality reduction ---


def test_no_reduction_when_well_conditioned() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_dimensionality_reduction

    r = recommend_dimensionality_reduction(condition_number=5.0, is_sparse=False, explained_variance=0.98)
    assert "No" in r["recommendation"] or "none" in r["sklearn_class"]


def test_pca_for_ill_conditioned_dense() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_dimensionality_reduction

    r = recommend_dimensionality_reduction(condition_number=100.0, is_sparse=False, explained_variance=0.80)
    assert "PCA" in r["recommendation"]


def test_truncated_svd_for_sparse() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_dimensionality_reduction

    r = recommend_dimensionality_reduction(condition_number=100.0, is_sparse=True, explained_variance=0.80)
    assert "TruncatedSVD" in r["recommendation"]
    assert r["confidence"] == "absolute"


# --- Section G: Hyperparameter ranges ---


def test_lasso_alpha_grid() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_hyperparameter_ranges

    r = recommend_hyperparameter_ranges(model_type="lasso", n=1000, p=50)
    assert "alpha" in r["recommendation"].lower() or "Log-space" in r["recommendation"]
    assert "ranges" in r
    assert r["ranges"]["alpha"]["low"] == pytest.approx(1e-5)


def test_gradient_boosting_ranges() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_hyperparameter_ranges

    r = recommend_hyperparameter_ranges(model_type="gradient_boosting", n=1000, p=50)
    assert "ranges" in r
    assert "learning_rate" in r["ranges"]
    assert "n_estimators" in r["ranges"]


# --- Section H: Cross-validation ---


def test_timeseries_split_for_temporal_data() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_cv_strategy

    r = recommend_cv_strategy(n=1000, is_classification=False, is_timeseries=True, has_groups=False)
    assert "TimeSeries" in r["recommendation"]
    assert r["confidence"] == "absolute"


def test_stratified_kfold_for_classification() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_cv_strategy

    r = recommend_cv_strategy(n=1000, is_classification=True, is_timeseries=False, has_groups=False)
    assert "Stratified" in r["recommendation"]


def test_loocv_for_small_datasets() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_cv_strategy

    r = recommend_cv_strategy(n=50, is_classification=False, is_timeseries=False, has_groups=False)
    assert "LeaveOneOut" in r["recommendation"] or "LOOCV" in r["recommendation"]


def test_group_kfold_for_grouped_data() -> None:
    from sciona.atoms.ml.model_selection.recommendations.atoms import recommend_cv_strategy

    r = recommend_cv_strategy(n=1000, is_classification=False, is_timeseries=False, has_groups=True)
    assert "Group" in r["recommendation"]


def test_all_recommendations_have_required_keys() -> None:
    """Verify all recommendation atoms return dicts with the required schema."""
    from sciona.atoms.ml.model_selection.recommendations.atoms import (
        recommend_regularization,
        recommend_loss_function,
        recommend_linear_model,
        recommend_tree_ensemble,
        recommend_preprocessing,
        recommend_dimensionality_reduction,
        recommend_hyperparameter_ranges,
        recommend_cv_strategy,
    )

    required_keys = {
        "recommendation", "confidence", "sklearn_class",
        "reasoning", "alternatives", "thresholds_applied", "source_sections",
    }

    results = [
        recommend_regularization(50.0, 5.0, 0.5, True),
        recommend_loss_function(0.5, 0.3),
        recommend_linear_model(1000, 50, False, 1.0, 1.0),
        recommend_tree_ensemble(500, 0, 0.1),
        recommend_preprocessing(np.array([0.2]), np.array([1.5]), False, True),
        recommend_dimensionality_reduction(5.0, False, 0.98),
        recommend_hyperparameter_ranges("lasso", 1000, 50),
        recommend_cv_strategy(200, True, False, False),
    ]

    for r in results:
        assert required_keys.issubset(set(r.keys())), f"Missing keys in {r['recommendation']}"
        assert r["confidence"] in {"high", "medium", "absolute"}
        assert len(r["source_sections"]) > 0
