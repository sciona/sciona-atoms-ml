"""Sklearn coordinate-descent CV subclass fit-return atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_subclass_fit_returns_super_result,
    witness_cd_cv_subclass_return_passthrough_required,
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


@register_atom(witness_cd_cv_subclass_return_passthrough_required)
@icontract.require(lambda cv_kind: _cv_kind(cv_kind), "cv_kind must be a known coordinate-descent CV subclass")
@icontract.ensure(
    lambda result, cv_kind: result is True,
    "coordinate-descent CV subclass fit wrappers return the super-fit result",
)
def cd_cv_subclass_return_passthrough_required(cv_kind: str) -> bool:
    """Return whether a coordinate-descent CV subclass fit wrapper returns super().fit."""
    del cv_kind
    return True


@register_atom(witness_cd_cv_subclass_fit_returns_super_result)
@icontract.require(lambda cv_kind: _cv_kind(cv_kind), "cv_kind must be a known coordinate-descent CV subclass")
@icontract.require(lambda super_fit_result: super_fit_result is not None, "super_fit_result must be provided")
@icontract.ensure(
    lambda result, cv_kind, super_fit_result: result is super_fit_result,
    "fit wrapper return must preserve the exact super().fit result identity",
)
def cd_cv_subclass_fit_returns_super_result(cv_kind: str, super_fit_result: object) -> object:
    """Return the exact object produced by the subclass super().fit call."""
    del cv_kind
    return super_fit_result
