"""Deterministic sklearn coordinate-descent multitask estimator shell atoms."""

from .atoms import (
    cd_multitask_dual_gap,
    cd_multitask_fit_return_self,
    cd_multitask_model_name,
    cd_multitask_mono_task_guard_required,
    cd_multitask_mono_task_message,
    cd_multitask_random_selection,
    cd_multitask_sparse_input_tag,
    cd_multitask_target_multi_output_tag,
    cd_multitask_target_single_output_tag,
)

__all__ = [
    "cd_multitask_model_name",
    "cd_multitask_mono_task_guard_required",
    "cd_multitask_mono_task_message",
    "cd_multitask_random_selection",
    "cd_multitask_dual_gap",
    "cd_multitask_fit_return_self",
    "cd_multitask_sparse_input_tag",
    "cd_multitask_target_multi_output_tag",
    "cd_multitask_target_single_output_tag",
]
