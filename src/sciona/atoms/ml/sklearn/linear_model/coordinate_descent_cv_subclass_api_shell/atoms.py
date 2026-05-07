"""Sklearn coordinate-descent CV subclass API-shell atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_subclass_estimator_name,
    witness_cd_cv_subclass_fit_forwards_sample_weight,
    witness_cd_cv_subclass_is_multitask,
    witness_cd_cv_subclass_path_name,
    witness_cd_cv_subclass_super_fit_args,
    witness_cd_cv_subclass_super_fit_kwargs,
    witness_cd_cv_subclass_target_single_output_tag,
)

_CV_KINDS = frozenset(
    {
        "lasso_cv",
        "elastic_net_cv",
        "multi_task_lasso_cv",
        "multi_task_elastic_net_cv",
    }
)


def _cv_kind(value: object) -> bool:
    return isinstance(value, str) and value in _CV_KINDS


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_cd_cv_subclass_path_name)
@icontract.require(lambda cv_kind: _cv_kind(cv_kind), "cv_kind must be a known coordinate-descent CV subclass")
@icontract.ensure(
    lambda result, cv_kind: result
    == (
        "lasso_path"
        if cv_kind in {"lasso_cv", "multi_task_lasso_cv"}
        else "enet_path"
    ),
    "CV subclass path helper must match sklearn staticmethod selection",
)
def cd_cv_subclass_path_name(cv_kind: str) -> str:
    """Return the path helper name selected by a coordinate-descent CV subclass."""
    return "lasso_path" if cv_kind in {"lasso_cv", "multi_task_lasso_cv"} else "enet_path"


@register_atom(witness_cd_cv_subclass_estimator_name)
@icontract.require(lambda cv_kind: _cv_kind(cv_kind), "cv_kind must be a known coordinate-descent CV subclass")
@icontract.ensure(
    lambda result, cv_kind: result
    == {
        "lasso_cv": "Lasso",
        "elastic_net_cv": "ElasticNet",
        "multi_task_lasso_cv": "MultiTaskLasso",
        "multi_task_elastic_net_cv": "MultiTaskElasticNet",
    }[cv_kind],
    "CV subclass estimator name must match _get_estimator()",
)
def cd_cv_subclass_estimator_name(cv_kind: str) -> str:
    """Return the concrete estimator name produced by a CV subclass."""
    return {
        "lasso_cv": "Lasso",
        "elastic_net_cv": "ElasticNet",
        "multi_task_lasso_cv": "MultiTaskLasso",
        "multi_task_elastic_net_cv": "MultiTaskElasticNet",
    }[cv_kind]


@register_atom(witness_cd_cv_subclass_is_multitask)
@icontract.require(lambda cv_kind: _cv_kind(cv_kind), "cv_kind must be a known coordinate-descent CV subclass")
@icontract.ensure(
    lambda result, cv_kind: _bool(result)
    and result == cv_kind.startswith("multi_task_"),
    "CV subclass multitask flag must match _is_multitask()",
)
def cd_cv_subclass_is_multitask(cv_kind: str) -> bool:
    """Return the boolean _is_multitask() value for a CV subclass."""
    return cv_kind.startswith("multi_task_")


@register_atom(witness_cd_cv_subclass_target_single_output_tag)
@icontract.require(lambda multitask: multitask is True, "single_output override is only used by multitask CV subclasses")
@icontract.ensure(
    lambda result: result is False,
    "multitask CV subclasses set target_tags.single_output to False",
)
def cd_cv_subclass_target_single_output_tag(multitask: bool) -> bool:
    """Return the target single_output tag override for multitask CV subclasses."""
    del multitask
    return False


@register_atom(witness_cd_cv_subclass_fit_forwards_sample_weight)
@icontract.require(lambda multitask: _bool(multitask), "multitask must be boolean")
@icontract.ensure(
    lambda result, multitask: _bool(result) and result == (not multitask),
    "sample_weight forwarding must be disabled only for multitask CV subclasses",
)
def cd_cv_subclass_fit_forwards_sample_weight(multitask: bool) -> bool:
    """Return whether subclass fit forwards sample_weight into super().fit."""
    return not multitask


@register_atom(witness_cd_cv_subclass_super_fit_args)
@icontract.ensure(
    lambda result, X, y: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is X
    and result[1] is y,
    "super().fit positional args must preserve X and y identity",
)
def cd_cv_subclass_super_fit_args(X: object, y: object) -> tuple[object, object]:
    """Return positional args passed into subclass super().fit(X, y, ...)."""
    return (X, y)


@register_atom(witness_cd_cv_subclass_super_fit_kwargs)
@icontract.require(lambda params: isinstance(params, dict), "params must be a dict")
@icontract.require(lambda forwards_sample_weight: _bool(forwards_sample_weight), "forwards_sample_weight must be boolean")
@icontract.ensure(
    lambda result, params, sample_weight, forwards_sample_weight: isinstance(result, dict)
    and all(result[key] is value for key, value in params.items())
    and (
        result.get("sample_weight") is sample_weight
        if forwards_sample_weight
        else "sample_weight" not in result
    ),
    "super().fit kwargs must preserve params and only add sample_weight when forwarded",
)
def cd_cv_subclass_super_fit_kwargs(
    params: dict[object, object],
    sample_weight: object,
    forwards_sample_weight: bool,
) -> dict[object, object]:
    """Return keyword args passed into subclass super().fit(...)."""
    result = dict(params)
    if forwards_sample_weight:
        result["sample_weight"] = sample_weight
    return result
