"""Ghost witnesses for sklearn coordinate-descent multitask estimator shell atoms."""

from __future__ import annotations


def witness_cd_multitask_model_name(has_l1_ratio_attr: object) -> object:
    """Describe the model-name shell in MultiTaskElasticNet.fit."""
    return has_l1_ratio_attr


def witness_cd_multitask_mono_task_guard_required(y_ndim: object) -> object:
    """Describe the mono-task output guard in MultiTaskElasticNet.fit."""
    return y_ndim


def witness_cd_multitask_mono_task_message(model_name: object) -> object:
    """Describe the mono-task ValueError message in MultiTaskElasticNet.fit."""
    return model_name


def witness_cd_multitask_random_selection(selection: object) -> object:
    """Describe the `self.selection == "random"` shell in MultiTaskElasticNet.fit."""
    return selection


def witness_cd_multitask_dual_gap(dual_gap: object, n_samples: object) -> object:
    """Describe the `self.dual_gap_ /= n_samples` shell in MultiTaskElasticNet.fit."""
    return dual_gap, n_samples


def witness_cd_multitask_fit_return_self(estimator_identity: object) -> object:
    """Describe the final `return self` shell in MultiTaskElasticNet.fit."""
    return estimator_identity


def witness_cd_multitask_sparse_input_tag(parent_sparse: object) -> object:
    """Describe the fixed sparse-input tag in MultiTaskElasticNet.__sklearn_tags__."""
    return parent_sparse


def witness_cd_multitask_target_multi_output_tag(parent_multi_output: object) -> object:
    """Describe the fixed multi_output tag in MultiTaskElasticNet.__sklearn_tags__."""
    return parent_multi_output


def witness_cd_multitask_target_single_output_tag(parent_single_output: object) -> object:
    """Describe the fixed single_output tag in MultiTaskElasticNet.__sklearn_tags__."""
    return parent_single_output
