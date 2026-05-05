"""Ghost witnesses for sklearn coordinate-descent CV API shell atoms."""

from __future__ import annotations


def witness_cd_cv_metadata_router_spec(class_name: object) -> object:
    """Describe the MetadataRouter owner and split mapping shell in LinearModelCV."""
    return class_name


def witness_cd_cv_multitask_bool(is_multitask_result: object) -> object:
    """Describe the `multitask = self._is_multitask()` shell in LinearModelCV.__sklearn_tags__."""
    return is_multitask_result


def witness_cd_cv_sparse_input_tag(multitask: object) -> object:
    """Describe the sparse-input tag shell in LinearModelCV.__sklearn_tags__."""
    return multitask


def witness_cd_cv_target_multi_output_tag(multitask: object) -> object:
    """Describe the multi_output tag shell in LinearModelCV.__sklearn_tags__."""
    return multitask


def witness_cd_cv_target_single_output_tag(multitask: object) -> object:
    """Describe the implied single_output tag shell for LinearModelCV-style multitask routing."""
    return multitask
