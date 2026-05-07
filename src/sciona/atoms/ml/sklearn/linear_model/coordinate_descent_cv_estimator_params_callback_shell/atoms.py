"""Sklearn coordinate-descent CV estimator parameter callback-shell atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_get_estimator_result,
    witness_cd_cv_model_get_params_result,
    witness_cd_cv_model_param_names,
    witness_cd_cv_path_get_params_result,
    witness_cd_cv_refit_get_params_result,
)


def _string_key_dict(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _shallow_copy_preserves_mapping(
    result: dict[str, object], params: dict[str, object]
) -> bool:
    return (
        result is not params
        and result.keys() == params.keys()
        and all(result[key] is params[key] for key in result)
    )


@register_atom(witness_cd_cv_get_estimator_result)
@icontract.require(lambda model: model is not None, "model must be present")
@icontract.ensure(
    lambda result, model: result is model,
    "_get_estimator callback result must preserve model identity",
)
def cd_cv_get_estimator_result(model: object) -> object:
    """Return the estimator produced by the deferred self._get_estimator() callback."""
    return model


@register_atom(witness_cd_cv_path_get_params_result)
@icontract.require(lambda self_params: _string_key_dict(self_params), "self_params must be a string-key dict")
@icontract.ensure(
    lambda result, self_params: _string_key_dict(result)
    and _shallow_copy_preserves_mapping(result, self_params),
    "path get_params result must preserve keys and value identities",
)
def cd_cv_path_get_params_result(self_params: dict[str, object]) -> dict[str, object]:
    """Return the self.get_params() mapping assigned to path_params."""
    return dict(self_params)


@register_atom(witness_cd_cv_refit_get_params_result)
@icontract.require(lambda self_params: _string_key_dict(self_params), "self_params must be a string-key dict")
@icontract.ensure(
    lambda result, self_params: _string_key_dict(result)
    and _shallow_copy_preserves_mapping(result, self_params),
    "refit get_params result must preserve keys and value identities",
)
def cd_cv_refit_get_params_result(self_params: dict[str, object]) -> dict[str, object]:
    """Return the self.get_params() mapping iterated during refit common-parameter setup."""
    return dict(self_params)


@register_atom(witness_cd_cv_model_get_params_result)
@icontract.require(lambda model_params: _string_key_dict(model_params), "model_params must be a string-key dict")
@icontract.ensure(
    lambda result, model_params: _string_key_dict(result)
    and _shallow_copy_preserves_mapping(result, model_params),
    "model get_params result must preserve keys and value identities",
)
def cd_cv_model_get_params_result(model_params: dict[str, object]) -> dict[str, object]:
    """Return the model.get_params() mapping used to filter common refit parameters."""
    return dict(model_params)


@register_atom(witness_cd_cv_model_param_names)
@icontract.require(lambda model_params: _string_key_dict(model_params), "model_params must be a string-key dict")
@icontract.ensure(
    lambda result, model_params: isinstance(result, set)
    and result == set(model_params.keys()),
    "model parameter names must match model.get_params() keys",
)
def cd_cv_model_param_names(model_params: dict[str, object]) -> set[str]:
    """Return the model parameter-name set used by common-parameter filtering."""
    return set(model_params.keys())
