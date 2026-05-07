"""Sklearn coordinate-descent ElasticNet API-shell atoms."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_elastic_net_init_attributes,
    witness_cd_elastic_net_path_name,
    witness_cd_elastic_net_sparse_decision_output,
    witness_cd_elastic_net_sparse_decision_required,
    witness_cd_elastic_net_sparse_dot_args,
    witness_cd_elastic_net_sparse_dot_kwargs,
    witness_cd_elastic_net_sparse_input_tag,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int_or_none(value: object) -> bool:
    return value is None or (isinstance(value, (int, np.integer)) and int(value) >= 1)


def _finite_coef(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim in {1, 2} and array.size >= 1 and np.all(np.isfinite(array)))


def _numeric_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.issubdtype(array.dtype, np.number))


@register_atom(witness_cd_elastic_net_path_name)
@icontract.require(lambda estimator_kind: estimator_kind == "elastic_net", "estimator_kind must be elastic_net")
@icontract.ensure(lambda result: result == "enet_path", "ElasticNet.path must be the enet_path helper")
def cd_elastic_net_path_name(estimator_kind: str) -> str:
    """Return the path helper name selected by ElasticNet."""
    del estimator_kind
    return "enet_path"


@register_atom(witness_cd_elastic_net_init_attributes)
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda max_iter: _positive_int_or_none(max_iter), "max_iter must be positive or None")
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda warm_start: _bool(warm_start), "warm_start must be boolean")
@icontract.require(lambda positive: _bool(positive), "positive must be boolean")
@icontract.require(lambda selection: isinstance(selection, str), "selection must be a string")
@icontract.ensure(
    lambda result, alpha, l1_ratio, fit_intercept, precompute, max_iter, copy_X, tol, warm_start, positive, random_state, selection: isinstance(result, dict)
    and set(result)
    == {
        "alpha",
        "l1_ratio",
        "fit_intercept",
        "precompute",
        "max_iter",
        "copy_X",
        "tol",
        "warm_start",
        "positive",
        "random_state",
        "selection",
    }
    and result["alpha"] is alpha
    and result["l1_ratio"] is l1_ratio
    and result["fit_intercept"] is fit_intercept
    and result["precompute"] is precompute
    and result["max_iter"] is max_iter
    and result["copy_X"] is copy_X
    and result["tol"] is tol
    and result["warm_start"] is warm_start
    and result["positive"] is positive
    and result["random_state"] is random_state
    and result["selection"] is selection,
    "ElasticNet init attributes must match sklearn assignment order and values",
)
def cd_elastic_net_init_attributes(
    alpha: object,
    l1_ratio: object,
    fit_intercept: bool,
    precompute: object,
    max_iter: int | None,
    copy_X: bool,
    tol: object,
    warm_start: bool,
    positive: bool,
    random_state: object,
    selection: str,
) -> dict[str, object]:
    """Return the attribute state assigned by ElasticNet.__init__."""
    return {
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "fit_intercept": fit_intercept,
        "precompute": precompute,
        "max_iter": max_iter,
        "copy_X": copy_X,
        "tol": tol,
        "warm_start": warm_start,
        "positive": positive,
        "random_state": random_state,
        "selection": selection,
    }


@register_atom(witness_cd_elastic_net_sparse_decision_required)
@icontract.require(lambda is_sparse: _bool(is_sparse), "is_sparse must be boolean")
@icontract.ensure(
    lambda result, is_sparse: _bool(result) and result is is_sparse,
    "sparse decision branch must match sparse.issparse(X)",
)
def cd_elastic_net_sparse_decision_required(is_sparse: bool) -> bool:
    """Return whether ElasticNet._decision_function uses the sparse branch."""
    return is_sparse


@register_atom(witness_cd_elastic_net_sparse_dot_args)
@icontract.require(lambda coef: _finite_coef(coef), "coef must be a finite 1D or 2D array")
@icontract.ensure(
    lambda result, X, coef: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is X
    and np.array_equal(result[1], np.asarray(coef).T),
    "sparse-dot args must preserve X and use coef_.T",
)
def cd_elastic_net_sparse_dot_args(
    X: object, coef: object
) -> tuple[object, NDArray[np.generic]]:
    """Return positional args for safe_sparse_dot(X, coef_.T, ...)."""
    return (X, np.asarray(coef).T)


@register_atom(witness_cd_elastic_net_sparse_dot_kwargs)
@icontract.require(lambda is_sparse: is_sparse is True, "sparse-dot kwargs are used only for sparse X")
@icontract.ensure(
    lambda result, is_sparse: isinstance(result, dict)
    and result == {"dense_output": True},
    "sparse-dot kwargs must request dense output",
)
def cd_elastic_net_sparse_dot_kwargs(is_sparse: bool) -> dict[str, bool]:
    """Return keyword args for safe_sparse_dot in the sparse branch."""
    del is_sparse
    return {"dense_output": True}


@register_atom(witness_cd_elastic_net_sparse_decision_output)
@icontract.require(lambda dot_output: _numeric_array(dot_output), "dot_output must be numeric")
@icontract.ensure(
    lambda result, dot_output, intercept: isinstance(result, np.ndarray)
    and np.array_equal(result, np.asarray(dot_output) + intercept),
    "sparse decision output must add intercept to sparse-dot output",
)
def cd_elastic_net_sparse_decision_output(
    dot_output: object, intercept: object
) -> NDArray[np.generic]:
    """Return safe_sparse_dot output plus the fitted intercept."""
    return np.asarray(dot_output) + intercept


@register_atom(witness_cd_elastic_net_sparse_input_tag)
@icontract.require(lambda parent_sparse=False: _bool(parent_sparse), "parent_sparse must be boolean")
@icontract.ensure(lambda result: result is True, "ElasticNet sparse-input tag must be True")
def cd_elastic_net_sparse_input_tag(parent_sparse: bool = False) -> bool:
    """Return the sparse-input tag set by ElasticNet.__sklearn_tags__."""
    del parent_sparse
    return True
