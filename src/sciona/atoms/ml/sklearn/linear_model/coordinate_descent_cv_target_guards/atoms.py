"""Sklearn coordinate-descent CV target-guard atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_multitask_monotask_guard_required,
    witness_cd_cv_multitask_monotask_message,
    witness_cd_cv_multitask_sparse_guard_required,
    witness_cd_cv_multitask_sparse_message,
    witness_cd_cv_non_multitask_guard_required,
    witness_cd_cv_non_multitask_message,
    witness_cd_cv_reference_preserving_validation_branch,
    witness_cd_cv_scalar_sample_weight_becomes_none,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _multitask_class_name(value: object) -> bool:
    return isinstance(value, str) and value.startswith("MultiTask") and len(value) > 9


@register_atom(witness_cd_cv_reference_preserving_validation_branch)
@icontract.require(lambda x_is_ndarray: _bool(x_is_ndarray), "x_is_ndarray must be boolean")
@icontract.require(lambda x_is_sparse: _bool(x_is_sparse), "x_is_sparse must be boolean")
@icontract.ensure(
    lambda result, x_is_ndarray, x_is_sparse: _bool(result)
    and result == (x_is_ndarray or x_is_sparse),
    "validation branch must match ndarray-or-sparse detection",
)
def cd_cv_reference_preserving_validation_branch(
    x_is_ndarray: bool, x_is_sparse: bool
) -> bool:
    """Return whether LinearModelCV.fit uses the reference-preserving X validation branch."""
    return x_is_ndarray or x_is_sparse


@register_atom(witness_cd_cv_non_multitask_guard_required)
@icontract.require(lambda multitask: _bool(multitask), "multitask must be boolean")
@icontract.require(lambda y_ndim: _positive_int(y_ndim), "y_ndim must be positive")
@icontract.require(lambda y_width: _positive_int(y_width), "y_width must be positive")
@icontract.ensure(
    lambda result, multitask, y_ndim, y_width: _bool(result)
    and result == ((not multitask) and int(y_ndim) > 1 and int(y_width) > 1),
    "non-multitask guard must match the sklearn multi-output condition",
)
def cd_cv_non_multitask_guard_required(
    multitask: bool, y_ndim: int, y_width: int
) -> bool:
    """Return whether LinearModelCV.fit should reject multi-output targets in non-multitask mode."""
    return (not multitask) and int(y_ndim) > 1 and int(y_width) > 1


@register_atom(witness_cd_cv_non_multitask_message)
@icontract.require(lambda class_name: _nonempty_str(class_name), "class_name must be nonempty")
@icontract.ensure(
    lambda result, class_name: isinstance(result, str)
    and result == ("For multi-task outputs, use MultiTask%s" % class_name),
    "non-multitask message must match sklearn formatting",
)
def cd_cv_non_multitask_message(class_name: str) -> str:
    """Return the non-multitask multi-output ValueError text used by LinearModelCV.fit."""
    return "For multi-task outputs, use MultiTask%s" % class_name


@register_atom(witness_cd_cv_multitask_sparse_guard_required)
@icontract.require(lambda multitask: _bool(multitask), "multitask must be boolean")
@icontract.require(lambda x_is_sparse: _bool(x_is_sparse), "x_is_sparse must be boolean")
@icontract.ensure(
    lambda result, multitask, x_is_sparse: _bool(result)
    and result == (multitask and x_is_sparse),
    "multitask sparse guard must match sklearn branching",
)
def cd_cv_multitask_sparse_guard_required(multitask: bool, x_is_sparse: bool) -> bool:
    """Return whether LinearModelCV.fit should reject sparse X in multitask mode."""
    return multitask and x_is_sparse


@register_atom(witness_cd_cv_multitask_sparse_message)
@icontract.require(lambda guard_required: _bool(guard_required), "guard_required must be boolean")
@icontract.ensure(
    lambda result: isinstance(result, str)
    and result == "X should be dense but a sparse matrix waspassed",
    "multitask sparse-input message must match sklearn formatting",
)
def cd_cv_multitask_sparse_message(guard_required: bool) -> str:
    """Return the multitask sparse-input TypeError text used by LinearModelCV.fit."""
    del guard_required
    return "X should be dense but a sparse matrix waspassed"


@register_atom(witness_cd_cv_multitask_monotask_guard_required)
@icontract.require(lambda multitask: _bool(multitask), "multitask must be boolean")
@icontract.require(lambda y_ndim: _positive_int(y_ndim), "y_ndim must be positive")
@icontract.ensure(
    lambda result, multitask, y_ndim: _bool(result)
    and result == (multitask and int(y_ndim) == 1),
    "multitask mono-task guard must match sklearn branching",
)
def cd_cv_multitask_monotask_guard_required(multitask: bool, y_ndim: int) -> bool:
    """Return whether LinearModelCV.fit should reject mono-task y in multitask mode."""
    return multitask and int(y_ndim) == 1


@register_atom(witness_cd_cv_multitask_monotask_message)
@icontract.require(
    lambda class_name: _multitask_class_name(class_name),
    "class_name must start with MultiTask",
)
@icontract.ensure(
    lambda result, class_name: isinstance(result, str)
    and result == ("For mono-task outputs, use %sCV" % class_name[9:]),
    "multitask mono-task message must match sklearn formatting",
)
def cd_cv_multitask_monotask_message(class_name: str) -> str:
    """Return the multitask mono-task ValueError text used by LinearModelCV.fit."""
    return "For mono-task outputs, use %sCV" % class_name[9:]


@register_atom(witness_cd_cv_scalar_sample_weight_becomes_none)
@icontract.require(
    lambda sample_weight_is_number: _bool(sample_weight_is_number),
    "sample_weight_is_number must be boolean",
)
@icontract.ensure(
    lambda result, sample_weight_is_number: _bool(result)
    and result == sample_weight_is_number,
    "scalar-sample-weight branch must match isinstance(sample_weight, numbers.Number)",
)
def cd_cv_scalar_sample_weight_becomes_none(sample_weight_is_number: bool) -> bool:
    """Return whether LinearModelCV.fit should replace sample_weight with None."""
    return sample_weight_is_number
