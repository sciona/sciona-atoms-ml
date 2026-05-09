"""Ghost witnesses for sklearn MultiTaskElasticNet fit-context atoms."""

from __future__ import annotations


def witness_cd_multitask_fit_context_kwargs(class_name: object) -> object:
    """Describe the `_fit_context` kwargs applied to MultiTaskElasticNet.fit."""
    return class_name


def witness_cd_multitask_fit_context_method_name(
    class_name: object, method_name: object
) -> object:
    """Describe the MultiTaskElasticNet method name decorated by `_fit_context`."""
    return class_name, method_name
