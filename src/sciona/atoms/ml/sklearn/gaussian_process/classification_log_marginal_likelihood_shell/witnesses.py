"""Ghost witnesses for Gaussian-process classification log-marginal-likelihood shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_gpc_log_marginal_likelihood_require_theta_for_gradient(
    theta_is_none: bool,
    eval_gradient: bool,
) -> None:
    """Describe the theta-required-for-gradient guard in GPC log-marginal likelihood."""
    del theta_is_none
    del eval_gradient
    return None


def witness_gpc_log_marginal_likelihood_cached_result(
    log_marginal_likelihood_value: float,
) -> float:
    """Describe the cached log-marginal-likelihood return value."""
    del log_marginal_likelihood_value
    return 0.0


def witness_gpc_log_marginal_likelihood_require_no_multiclass_gradient(
    n_classes: int,
    eval_gradient: bool,
) -> None:
    """Describe the multiclass gradient guard in GPC log-marginal likelihood."""
    del n_classes
    del eval_gradient
    return None


def witness_gpc_log_marginal_likelihood_use_binary_branch(
    n_classes: int,
) -> bool:
    """Describe whether GPC log-marginal likelihood uses the binary branch."""
    del n_classes
    return False


def witness_gpc_log_marginal_likelihood_use_shared_theta(
    theta: AbstractArray,
    n_dims: int,
    n_classes: int,
) -> bool:
    """Describe whether GPC log-marginal likelihood uses one shared theta across multiclass estimators."""
    del theta
    del n_dims
    del n_classes
    return False


def witness_gpc_log_marginal_likelihood_use_compound_theta(
    theta: AbstractArray,
    n_dims: int,
    n_classes: int,
) -> bool:
    """Describe whether GPC log-marginal likelihood uses a concatenated theta vector across multiclass estimators."""
    del theta
    del n_dims
    del n_classes
    return False


def witness_gpc_log_marginal_likelihood_theta_slice(
    theta: AbstractArray,
    n_dims: int,
    estimator_index: int,
) -> AbstractArray:
    """Describe one multiclass theta slice extracted for a sub-estimator."""
    del theta
    del estimator_index
    return AbstractArray(shape=(int(n_dims),), dtype="float64")


def witness_gpc_log_marginal_likelihood_mean(
    values: AbstractArray,
) -> float:
    """Describe the scalar mean of multiclass log-marginal-likelihood callback outputs."""
    del values
    return 0.0


def witness_gpc_log_marginal_likelihood_theta_shape_message(
    n_dims: int,
    n_classes: int,
    theta_size: int,
) -> str:
    """Describe the invalid theta-shape message for multiclass GPC."""
    del n_dims
    del n_classes
    del theta_size
    return ""
