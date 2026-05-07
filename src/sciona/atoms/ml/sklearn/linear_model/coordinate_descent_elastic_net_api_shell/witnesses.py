"""Ghost witnesses for sklearn ElasticNet API atoms."""

from __future__ import annotations


def witness_cd_elastic_net_path_name(estimator_kind: object) -> object:
    """Describe ElasticNet path helper selection."""
    return estimator_kind


def witness_cd_elastic_net_init_attributes(
    alpha: object,
    l1_ratio: object,
    fit_intercept: object,
    precompute: object,
    max_iter: object,
    copy_X: object,
    tol: object,
    warm_start: object,
    positive: object,
    random_state: object,
    selection: object,
) -> object:
    """Describe attributes assigned by ElasticNet.__init__."""
    return (
        alpha,
        l1_ratio,
        fit_intercept,
        precompute,
        max_iter,
        copy_X,
        tol,
        warm_start,
        positive,
        random_state,
        selection,
    )


def witness_cd_elastic_net_sparse_decision_required(is_sparse: object) -> object:
    """Describe the sparse-input branch in ElasticNet._decision_function."""
    return is_sparse


def witness_cd_elastic_net_sparse_dot_args(X: object, coef: object) -> object:
    """Describe sparse-dot positional arguments for ElasticNet prediction."""
    return X, coef


def witness_cd_elastic_net_sparse_dot_kwargs(is_sparse: object) -> object:
    """Describe sparse-dot keyword arguments for ElasticNet prediction."""
    return is_sparse


def witness_cd_elastic_net_sparse_decision_output(
    dot_output: object, intercept: object
) -> object:
    """Describe adding the fitted intercept to sparse-dot output."""
    return dot_output, intercept


def witness_cd_elastic_net_sparse_input_tag(parent_sparse: object) -> object:
    """Describe the sparse-input tag override."""
    return parent_sparse
