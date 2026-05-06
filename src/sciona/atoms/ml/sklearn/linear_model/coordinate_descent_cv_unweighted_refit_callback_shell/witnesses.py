"""Ghost witnesses for sklearn coordinate-descent CV unweighted refit callback-shell atoms."""

from __future__ import annotations


def witness_cd_cv_refit_unweighted_fit_call_required(sample_weight: object) -> object:
    """Describe the sample_weight is None branch for unweighted LinearModelCV refit."""
    return sample_weight


def witness_cd_cv_refit_unweighted_fit_args(X: object, y: object, sample_weight: object) -> object:
    """Describe the positional model.fit(X, y) shell in LinearModelCV.fit."""
    return X, y, sample_weight


def witness_cd_cv_refit_unweighted_fit_kwargs(sample_weight: object) -> object:
    """Describe the absence of kwargs for model.fit(X, y) in LinearModelCV.fit."""
    return sample_weight


def witness_cd_cv_refit_unweighted_fitted_model(model_after_fit: object) -> object:
    """Describe the fitted model object after unweighted model.fit(X, y)."""
    return model_after_fit
