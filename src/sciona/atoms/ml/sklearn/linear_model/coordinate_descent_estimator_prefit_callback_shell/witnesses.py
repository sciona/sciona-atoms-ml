"""Ghost witnesses for sklearn coordinate-descent estimator pre-fit callback atoms."""

from __future__ import annotations


def witness_cd_estimator_prefit_result_unpack(prefit_result: object) -> object:
    """Describe fields retained from the ElasticNet.fit _pre_fit callback result."""
    return prefit_result


def witness_cd_estimator_set_order_result_unpack(order_result: object) -> object:
    """Describe fields retained from the ElasticNet.fit _set_order callback result."""
    return order_result


def witness_cd_estimator_prefit_xy_payload(
    X: object,
    y: object,
    X_offset: object,
    y_offset: object,
    X_scale: object,
    precompute: object,
    Xy: object,
) -> object:
    """Describe the named post-_pre_fit payload consumed by later estimator fitting."""
    return X, y, X_offset, y_offset, X_scale, precompute, Xy
