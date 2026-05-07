"""Ghost witnesses for sklearn coordinate-descent CV estimator parameter callback shells."""

from __future__ import annotations


def witness_cd_cv_get_estimator_result(model: object) -> object:
    """Describe the estimator returned by self._get_estimator()."""
    return model


def witness_cd_cv_path_get_params_result(self_params: object) -> object:
    """Describe the self.get_params() mapping used for path_params."""
    return self_params


def witness_cd_cv_refit_get_params_result(self_params: object) -> object:
    """Describe the self.get_params() mapping used during refit setup."""
    return self_params


def witness_cd_cv_model_get_params_result(model_params: object) -> object:
    """Describe the model.get_params() mapping used during refit setup."""
    return model_params


def witness_cd_cv_model_param_names(model_params: object) -> object:
    """Describe model parameter names derived from model.get_params()."""
    return model_params
