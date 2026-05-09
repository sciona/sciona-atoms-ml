"""Sklearn LinearModelCV abstract API atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_base_abstract_method_names,
    witness_cd_cv_base_abstract_method_roles,
    witness_cd_cv_base_path_signature_payload,
)


_ABSTRACT_METHODS = ("_get_estimator", "_is_multitask", "path")
_ABSTRACT_ROLES = {
    "_get_estimator": "refit_estimator_factory",
    "_is_multitask": "target_shape_policy",
    "path": "coordinate_descent_path_callable",
}


@register_atom(witness_cd_cv_base_abstract_method_names)
@icontract.require(lambda class_name: class_name == "LinearModelCV", "class_name must be LinearModelCV")
@icontract.ensure(
    lambda result: isinstance(result, tuple) and result == _ABSTRACT_METHODS,
    "LinearModelCV abstract methods must match sklearn declaration order",
)
def cd_cv_base_abstract_method_names(class_name: str) -> tuple[str, ...]:
    """Return the abstract method names declared by LinearModelCV."""
    del class_name
    return _ABSTRACT_METHODS


@register_atom(witness_cd_cv_base_abstract_method_roles)
@icontract.require(lambda method_name: method_name in _ABSTRACT_METHODS, "method_name must be a LinearModelCV abstract method")
@icontract.ensure(
    lambda result, method_name: isinstance(result, str) and result == _ABSTRACT_ROLES[method_name],
    "abstract method role must match the LinearModelCV base contract",
)
def cd_cv_base_abstract_method_roles(method_name: str) -> str:
    """Return the role of a LinearModelCV abstract method."""
    return _ABSTRACT_ROLES[method_name]


@register_atom(witness_cd_cv_base_path_signature_payload)
@icontract.require(lambda kwargs: isinstance(kwargs, Mapping), "kwargs must be a mapping")
@icontract.ensure(
    lambda result, X, y, kwargs: isinstance(result, dict)
    and result["X"] is X
    and result["y"] is y
    and result["kwargs"] == dict(kwargs),
    "path signature payload must preserve X, y, and **kwargs",
)
def cd_cv_base_path_signature_payload(
    X: object,
    y: object,
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Return the payload shape accepted by LinearModelCV.path(X, y, **kwargs)."""
    return {"X": X, "y": y, "kwargs": dict(kwargs)}
