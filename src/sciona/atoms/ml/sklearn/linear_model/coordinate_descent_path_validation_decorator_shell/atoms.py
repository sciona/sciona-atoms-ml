"""Sklearn coordinate-descent path validation-decorator atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_validation_param_names,
    witness_cd_lasso_path_validation_param_names,
    witness_cd_path_validation_param_descriptors,
    witness_cd_path_validation_prefers_skip_nested,
)


_LASSO_PATH_VALIDATION_PARAM_NAMES = (
    "X",
    "y",
    "eps",
    "n_alphas",
    "alphas",
    "precompute",
    "Xy",
    "copy_X",
    "coef_init",
    "verbose",
    "return_n_iter",
    "positive",
)

_ENET_PATH_VALIDATION_PARAM_NAMES = (
    "X",
    "y",
    "l1_ratio",
    "eps",
    "n_alphas",
    "alphas",
    "precompute",
    "Xy",
    "copy_X",
    "coef_init",
    "verbose",
    "return_n_iter",
    "positive",
    "check_input",
)

_PATH_VALIDATION_DESCRIPTORS = {
    "lasso_path": {
        "X": ("array-like", "sparse matrix"),
        "y": ("array-like", "sparse matrix"),
        "eps": (("interval", "Real", 0, None, "neither"),),
        "n_alphas": (("interval", "Integral", 1, None, "left"),),
        "alphas": ("array-like", None),
        "precompute": (("str_options", ("auto",)), "boolean", "array-like"),
        "Xy": ("array-like", None),
        "copy_X": ("boolean",),
        "coef_init": ("array-like", None),
        "verbose": ("verbose",),
        "return_n_iter": ("boolean",),
        "positive": ("boolean",),
    },
    "enet_path": {
        "X": ("array-like", "sparse matrix"),
        "y": ("array-like", "sparse matrix"),
        "l1_ratio": (("interval", "Real", 0.0, 1.0, "both"),),
        "eps": (("interval", "Real", 0.0, None, "neither"),),
        "n_alphas": (("interval", "Integral", 1, None, "left"),),
        "alphas": ("array-like", None),
        "precompute": (("str_options", ("auto",)), "boolean", "array-like"),
        "Xy": ("array-like", None),
        "copy_X": ("boolean",),
        "coef_init": ("array-like", None),
        "verbose": ("verbose",),
        "return_n_iter": ("boolean",),
        "positive": ("boolean",),
        "check_input": ("boolean",),
    },
}


def _path_name(value: object) -> bool:
    return isinstance(value, str) and value in _PATH_VALIDATION_DESCRIPTORS


@register_atom(witness_cd_lasso_path_validation_param_names)
@icontract.require(lambda path_name: path_name == "lasso_path", "path_name must be lasso_path")
@icontract.ensure(
    lambda result: isinstance(result, tuple)
    and result == _LASSO_PATH_VALIDATION_PARAM_NAMES,
    "lasso_path validate_params names must match sklearn declaration order",
)
def cd_lasso_path_validation_param_names(path_name: str) -> tuple[str, ...]:
    """Return lasso_path validate_params keys in declaration order."""
    del path_name
    return _LASSO_PATH_VALIDATION_PARAM_NAMES


@register_atom(witness_cd_enet_path_validation_param_names)
@icontract.require(lambda path_name: path_name == "enet_path", "path_name must be enet_path")
@icontract.ensure(
    lambda result: isinstance(result, tuple)
    and result == _ENET_PATH_VALIDATION_PARAM_NAMES,
    "enet_path validate_params names must match sklearn declaration order",
)
def cd_enet_path_validation_param_names(path_name: str) -> tuple[str, ...]:
    """Return enet_path validate_params keys in declaration order."""
    del path_name
    return _ENET_PATH_VALIDATION_PARAM_NAMES


@register_atom(witness_cd_path_validation_param_descriptors)
@icontract.require(lambda path_name: _path_name(path_name), "path_name must be lasso_path or enet_path")
@icontract.ensure(
    lambda result, path_name: isinstance(result, dict)
    and result == _PATH_VALIDATION_DESCRIPTORS[path_name]
    and tuple(result)
    == (
        _LASSO_PATH_VALIDATION_PARAM_NAMES
        if path_name == "lasso_path"
        else _ENET_PATH_VALIDATION_PARAM_NAMES
    ),
    "validation descriptors must match the selected path helper",
)
def cd_path_validation_param_descriptors(path_name: str) -> dict[str, tuple[object, ...]]:
    """Return compact descriptors for a path helper validate_params schema."""
    return dict(_PATH_VALIDATION_DESCRIPTORS[path_name])


@register_atom(witness_cd_path_validation_prefers_skip_nested)
@icontract.require(lambda path_name: _path_name(path_name), "path_name must be lasso_path or enet_path")
@icontract.ensure(
    lambda result: result is True,
    "path helper validate_params decorators prefer skipping nested validation",
)
def cd_path_validation_prefers_skip_nested(path_name: str) -> bool:
    """Return the shared prefer_skip_nested_validation flag for path helpers."""
    del path_name
    return True
