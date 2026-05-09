"""Sklearn LinearModelCV fit-context atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_base_fit_context_kwargs,
    witness_cd_cv_base_fit_context_method_name,
)


@register_atom(witness_cd_cv_base_fit_context_kwargs)
@icontract.require(lambda method_name: method_name == "fit", "method_name must be fit")
@icontract.ensure(
    lambda result: isinstance(result, dict)
    and result == {"prefer_skip_nested_validation": True},
    "LinearModelCV.fit context kwargs must match sklearn decorator payload",
)
def cd_cv_base_fit_context_kwargs(method_name: str) -> dict[str, bool]:
    """Return kwargs passed to `_fit_context` for LinearModelCV.fit."""
    del method_name
    return {"prefer_skip_nested_validation": True}


@register_atom(witness_cd_cv_base_fit_context_method_name)
@icontract.require(
    lambda decorated_callable_name: isinstance(decorated_callable_name, str)
    and len(decorated_callable_name) >= 1,
    "decorated_callable_name must be a nonempty string",
)
@icontract.ensure(
    lambda result, decorated_callable_name: isinstance(result, str)
    and result == decorated_callable_name
    and result == "fit",
    "decorated callable name must be LinearModelCV.fit",
)
def cd_cv_base_fit_context_method_name(decorated_callable_name: str) -> str:
    """Return the method name receiving the LinearModelCV `_fit_context` wrapper."""
    return decorated_callable_name
