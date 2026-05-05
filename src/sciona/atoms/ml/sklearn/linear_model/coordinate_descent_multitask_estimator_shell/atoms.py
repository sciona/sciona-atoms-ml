"""Sklearn coordinate-descent multitask estimator shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_multitask_dual_gap,
    witness_cd_multitask_fit_return_self,
    witness_cd_multitask_model_name,
    witness_cd_multitask_mono_task_guard_required,
    witness_cd_multitask_mono_task_message,
    witness_cd_multitask_random_selection,
    witness_cd_multitask_sparse_input_tag,
    witness_cd_multitask_target_multi_output_tag,
    witness_cd_multitask_target_single_output_tag,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_cd_multitask_model_name)
@icontract.require(
    lambda has_l1_ratio_attr: isinstance(has_l1_ratio_attr, bool),
    "has_l1_ratio_attr must be boolean",
)
@icontract.ensure(
    lambda result, has_l1_ratio_attr: isinstance(result, str)
    and result == ("ElasticNet" if has_l1_ratio_attr else "Lasso"),
    "model name must match the estimator-shell branch",
)
def cd_multitask_model_name(has_l1_ratio_attr: bool) -> str:
    """Return the model name used in the mono-task guard message."""
    return "ElasticNet" if has_l1_ratio_attr else "Lasso"


@register_atom(witness_cd_multitask_mono_task_guard_required)
@icontract.require(lambda y_ndim: _positive_int(y_ndim), "y_ndim must be positive")
@icontract.ensure(
    lambda result, y_ndim: isinstance(result, bool) and result == (int(y_ndim) == 1),
    "mono-task guard must match y.ndim == 1",
)
def cd_multitask_mono_task_guard_required(y_ndim: int) -> bool:
    """Return whether MultiTaskElasticNet.fit should raise for mono-task targets."""
    return int(y_ndim) == 1


@register_atom(witness_cd_multitask_mono_task_message)
@icontract.require(
    lambda model_name: isinstance(model_name, str) and model_name in {"ElasticNet", "Lasso"},
    "model_name must be ElasticNet or Lasso",
)
@icontract.ensure(
    lambda result, model_name: isinstance(result, str)
    and result == ("For mono-task outputs, use %s" % model_name),
    "mono-task message must match sklearn formatting",
)
def cd_multitask_mono_task_message(model_name: str) -> str:
    """Return the mono-task ValueError message used by MultiTaskElasticNet.fit."""
    return "For mono-task outputs, use %s" % model_name


@register_atom(witness_cd_multitask_random_selection)
@icontract.require(
    lambda selection: isinstance(selection, str) and selection in {"random", "cyclic"},
    "selection must be random or cyclic",
)
@icontract.ensure(
    lambda result, selection: isinstance(result, bool) and result == (selection == "random"),
    "random-selection flag must match selection == 'random'",
)
def cd_multitask_random_selection(selection: str) -> bool:
    """Return whether the multitask estimator should use random selection."""
    return selection == "random"


@register_atom(witness_cd_multitask_dual_gap)
@icontract.require(lambda dual_gap: np.isfinite(float(dual_gap)), "dual_gap must be finite")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be positive")
@icontract.ensure(
    lambda result, dual_gap, n_samples: np.isfinite(float(result))
    and np.isclose(float(result), float(dual_gap) / int(n_samples)),
    "scaled dual gap must equal dual_gap / n_samples",
)
def cd_multitask_dual_gap(dual_gap: float, n_samples: int) -> float:
    """Return the public dual gap after multitask solver scaling correction."""
    return float(dual_gap) / int(n_samples)


@register_atom(witness_cd_multitask_fit_return_self)
@icontract.ensure(
    lambda result, estimator_identity: result is estimator_identity,
    "fit return must pass self through unchanged",
)
def cd_multitask_fit_return_self(estimator_identity: object) -> object:
    """Return self from MultiTaskElasticNet.fit."""
    return estimator_identity


@register_atom(witness_cd_multitask_sparse_input_tag)
@icontract.require(lambda parent_sparse=False: _bool(parent_sparse), "parent_sparse must be boolean")
@icontract.ensure(lambda result: _bool(result) and result is False, "sparse-input tag must be False")
def cd_multitask_sparse_input_tag(parent_sparse: bool = False) -> bool:
    """Return the fixed sparse-input tag used by MultiTaskElasticNet.__sklearn_tags__."""
    del parent_sparse
    return False


@register_atom(witness_cd_multitask_target_multi_output_tag)
@icontract.require(
    lambda parent_multi_output=False: _bool(parent_multi_output),
    "parent_multi_output must be boolean",
)
@icontract.ensure(lambda result: _bool(result) and result is True, "multi_output tag must be True")
def cd_multitask_target_multi_output_tag(parent_multi_output: bool = False) -> bool:
    """Return the fixed multi_output tag used by MultiTaskElasticNet.__sklearn_tags__."""
    del parent_multi_output
    return True


@register_atom(witness_cd_multitask_target_single_output_tag)
@icontract.require(
    lambda parent_single_output=False: _bool(parent_single_output),
    "parent_single_output must be boolean",
)
@icontract.ensure(lambda result: _bool(result) and result is False, "single_output tag must be False")
def cd_multitask_target_single_output_tag(parent_single_output: bool = False) -> bool:
    """Return the fixed single_output tag used by MultiTaskElasticNet.__sklearn_tags__."""
    del parent_single_output
    return False
