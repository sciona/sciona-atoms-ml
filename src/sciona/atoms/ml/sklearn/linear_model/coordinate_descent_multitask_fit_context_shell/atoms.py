"""Sklearn MultiTaskElasticNet fit-context atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_multitask_fit_context_kwargs,
    witness_cd_multitask_fit_context_method_name,
)


@register_atom(witness_cd_multitask_fit_context_kwargs)
@icontract.require(lambda class_name: class_name == "MultiTaskElasticNet", "class_name must be MultiTaskElasticNet")
@icontract.ensure(
    lambda result: isinstance(result, dict)
    and result == {"prefer_skip_nested_validation": True},
    "MultiTaskElasticNet.fit context kwargs must match sklearn decorator payload",
)
def cd_multitask_fit_context_kwargs(class_name: str) -> dict[str, bool]:
    """Return kwargs passed to `_fit_context` for MultiTaskElasticNet.fit."""
    del class_name
    return {"prefer_skip_nested_validation": True}


@register_atom(witness_cd_multitask_fit_context_method_name)
@icontract.require(lambda class_name: class_name == "MultiTaskElasticNet", "class_name must be MultiTaskElasticNet")
@icontract.require(
    lambda decorated_callable_name: isinstance(decorated_callable_name, str)
    and len(decorated_callable_name) >= 1,
    "decorated_callable_name must be a nonempty string",
)
@icontract.ensure(
    lambda result, decorated_callable_name: isinstance(result, str)
    and result == decorated_callable_name
    and result == "fit",
    "decorated callable name must be MultiTaskElasticNet.fit",
)
def cd_multitask_fit_context_method_name(
    class_name: str, decorated_callable_name: str
) -> str:
    """Return the method name receiving the MultiTaskElasticNet `_fit_context` wrapper."""
    del class_name
    return decorated_callable_name
