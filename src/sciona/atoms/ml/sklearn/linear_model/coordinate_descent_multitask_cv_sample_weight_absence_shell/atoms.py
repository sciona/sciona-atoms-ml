"""Sklearn multitask coordinate-descent CV sample-weight absence atoms."""

from __future__ import annotations

from collections.abc import Iterable

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_multitask_cv_fit_params_name,
    witness_cd_multitask_cv_fit_signature,
    witness_cd_multitask_cv_fit_signature_classes,
    witness_cd_multitask_cv_sample_weight_absent,
)


_MULTITASK_CV_CLASSES = frozenset({"MultiTaskElasticNetCV", "MultiTaskLassoCV"})
_FIT_PARAMETER_NAMES = ("self", "X", "y", "params")


def _multitask_cv_class(value: object) -> bool:
    return isinstance(value, str) and value in _MULTITASK_CV_CLASSES


def _string_iterable(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


@register_atom(witness_cd_multitask_cv_fit_signature)
@icontract.require(
    lambda class_name: _multitask_cv_class(class_name),
    "class_name must be a multitask coordinate-descent CV class",
)
@icontract.ensure(
    lambda result: isinstance(result, tuple) and result == _FIT_PARAMETER_NAMES,
    "multitask CV fit signature must be (self, X, y, **params)",
)
def cd_multitask_cv_fit_signature(class_name: str) -> tuple[str, str, str, str]:
    """Return the parameter names in the multitask CV fit signature."""
    del class_name
    return _FIT_PARAMETER_NAMES


@register_atom(witness_cd_multitask_cv_fit_params_name)
@icontract.require(lambda params_name: isinstance(params_name, str), "params_name must be a string")
@icontract.ensure(
    lambda result, params_name: isinstance(result, bool) and result == (params_name == "params"),
    "multitask CV fit variadic keyword parameter must be named params",
)
def cd_multitask_cv_fit_params_name(params_name: str) -> bool:
    """Return whether the variadic keyword parameter is named params."""
    return params_name == "params"


@register_atom(witness_cd_multitask_cv_sample_weight_absent)
@icontract.require(lambda parameter_names: _string_iterable(parameter_names), "parameter_names must be an iterable")
@icontract.ensure(
    lambda result, parameter_names: isinstance(result, bool)
    and result == ("sample_weight" not in tuple(parameter_names)),
    "sample_weight must be absent from the multitask CV fit signature",
)
def cd_multitask_cv_sample_weight_absent(parameter_names: Iterable[str]) -> bool:
    """Return whether sample_weight is absent from a fit signature."""
    return "sample_weight" not in tuple(parameter_names)


@register_atom(witness_cd_multitask_cv_fit_signature_classes)
@icontract.require(lambda class_names: _string_iterable(class_names), "class_names must be an iterable")
@icontract.ensure(
    lambda result, class_names: isinstance(result, tuple)
    and result == ("MultiTaskElasticNetCV", "MultiTaskLassoCV")
    and set(result).issubset(set(class_names)),
    "signature seam must cover both multitask coordinate-descent CV wrappers",
)
def cd_multitask_cv_fit_signature_classes(class_names: Iterable[str]) -> tuple[str, str]:
    """Return the multitask CV classes sharing the sample-weight absence seam."""
    available = set(class_names)
    if not _MULTITASK_CV_CLASSES.issubset(available):
        missing = ", ".join(sorted(_MULTITASK_CV_CLASSES - available))
        raise ValueError(f"missing multitask CV classes: {missing}")
    return ("MultiTaskElasticNetCV", "MultiTaskLassoCV")
