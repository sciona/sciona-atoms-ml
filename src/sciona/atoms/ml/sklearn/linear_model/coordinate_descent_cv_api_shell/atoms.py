"""Sklearn coordinate-descent CV API shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_metadata_router_spec,
    witness_cd_cv_multitask_bool,
    witness_cd_cv_sparse_input_tag,
    witness_cd_cv_target_multi_output_tag,
    witness_cd_cv_target_single_output_tag,
)


@register_atom(witness_cd_cv_metadata_router_spec)
@icontract.require(
    lambda class_name: isinstance(class_name, str) and len(class_name) >= 1,
    "class_name must be a nonempty string",
)
@icontract.ensure(
    lambda result, class_name: result
    == {
        "owner": class_name,
        "caller": "fit",
        "callee": "split",
    },
    "metadata router spec must match the LinearModelCV routing shell",
)
def cd_cv_metadata_router_spec(class_name: str) -> dict[str, str]:
    """Return the metadata-routing spec assembled by LinearModelCV.get_metadata_routing."""
    return {
        "owner": class_name,
        "caller": "fit",
        "callee": "split",
    }


@register_atom(witness_cd_cv_multitask_bool)
@icontract.require(
    lambda is_multitask_result: isinstance(is_multitask_result, bool),
    "_is_multitask() result must be boolean",
)
@icontract.ensure(
    lambda result, is_multitask_result: isinstance(result, bool) and result == is_multitask_result,
    "multitask bool must pass through _is_multitask()",
)
def cd_cv_multitask_bool(is_multitask_result: bool) -> bool:
    """Return the multitask boolean used by LinearModelCV.__sklearn_tags__."""
    return is_multitask_result


@register_atom(witness_cd_cv_sparse_input_tag)
@icontract.require(lambda multitask: isinstance(multitask, bool), "multitask must be boolean")
@icontract.ensure(
    lambda result, multitask: isinstance(result, bool) and result == (not multitask),
    "sparse-input tag must match not multitask",
)
def cd_cv_sparse_input_tag(multitask: bool) -> bool:
    """Return the sparse-input tag used by LinearModelCV.__sklearn_tags__."""
    return not multitask


@register_atom(witness_cd_cv_target_multi_output_tag)
@icontract.require(lambda multitask: isinstance(multitask, bool), "multitask must be boolean")
@icontract.ensure(
    lambda result, multitask: isinstance(result, bool) and result == multitask,
    "target multi_output tag must match multitask",
)
def cd_cv_target_multi_output_tag(multitask: bool) -> bool:
    """Return the target multi_output tag used by LinearModelCV.__sklearn_tags__."""
    return multitask


@register_atom(witness_cd_cv_target_single_output_tag)
@icontract.require(lambda multitask: isinstance(multitask, bool), "multitask must be boolean")
@icontract.ensure(
    lambda result, multitask: isinstance(result, bool) and result == (not multitask),
    "target single_output tag must be the complement of multitask",
)
def cd_cv_target_single_output_tag(multitask: bool) -> bool:
    """Return the implied target single_output tag complement for LinearModelCV-style shells."""
    return not multitask
