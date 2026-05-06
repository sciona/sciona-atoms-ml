"""Ghost witnesses for sklearn coordinate-descent CV refit-setup atoms."""

from __future__ import annotations


def witness_cd_cv_refit_common_params(self_params: object, model_param_names: object) -> object:
    """Describe the common-params filtering shell in LinearModelCV.fit."""
    return self_params, model_param_names


def witness_cd_cv_refit_model_alpha(best_alpha: object) -> object:
    """Describe the selected alpha assignment shell in LinearModelCV.fit."""
    return best_alpha


def witness_cd_cv_refit_model_l1_ratio(best_l1_ratio: object) -> object:
    """Describe the selected l1_ratio assignment shell in LinearModelCV.fit."""
    return best_l1_ratio


def witness_cd_cv_refit_copy_x(copy_x: object) -> object:
    """Describe the model.copy_X assignment shell in LinearModelCV.fit."""
    return copy_x


def witness_cd_cv_refit_precompute_auto_guard_required(precompute: object) -> object:
    """Describe the precompute=='auto' guard in LinearModelCV.fit."""
    return precompute


def witness_cd_cv_refit_precompute_value(
    precompute: object, precompute_auto_guard_required: object
) -> object:
    """Describe the resolved model.precompute value in LinearModelCV.fit."""
    return precompute, precompute_auto_guard_required


def witness_cd_cv_refit_fit_call_uses_sample_weight(sample_weight: object) -> object:
    """Describe the sample-weight fit-dispatch branch in LinearModelCV.fit."""
    return sample_weight
