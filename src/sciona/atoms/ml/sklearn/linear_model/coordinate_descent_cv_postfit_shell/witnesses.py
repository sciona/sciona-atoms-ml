"""Ghost witnesses for sklearn coordinate-descent CV post-fit shell atoms."""

from __future__ import annotations


def witness_cd_cv_refit_uses_sample_weight(sample_weight: object) -> object:
    """Describe the sample-weight refit branch in LinearModelCV.fit."""
    return sample_weight


def witness_cd_cv_delete_l1_ratio_required(has_l1_ratio_attr: object) -> object:
    """Describe the `if not hasattr(self, "l1_ratio")` cleanup branch in LinearModelCV.fit."""
    return has_l1_ratio_attr


def witness_cd_cv_fit_coef(model_coef: object) -> object:
    """Describe the `self.coef_ = model.coef_` shell in LinearModelCV.fit."""
    return model_coef


def witness_cd_cv_fit_intercept(model_intercept: object) -> object:
    """Describe the `self.intercept_ = model.intercept_` shell in LinearModelCV.fit."""
    return model_intercept


def witness_cd_cv_fit_dual_gap(model_dual_gap: object) -> object:
    """Describe the `self.dual_gap_ = model.dual_gap_` shell in LinearModelCV.fit."""
    return model_dual_gap


def witness_cd_cv_fit_n_iter(model_n_iter: object) -> object:
    """Describe the `self.n_iter_ = model.n_iter_` shell in LinearModelCV.fit."""
    return model_n_iter


def witness_cd_cv_fit_return_self(estimator_identity: object) -> object:
    """Describe the final `return self` shell in LinearModelCV.fit."""
    return estimator_identity
