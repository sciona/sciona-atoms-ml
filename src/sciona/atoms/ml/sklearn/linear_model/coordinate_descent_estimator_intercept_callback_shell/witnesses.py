"""Ghost witnesses for sklearn coordinate-descent estimator intercept callback atoms."""

from __future__ import annotations


def witness_cd_estimator_set_intercept_args(
    X_offset: object,
    y_offset: object,
    X_scale: object,
) -> object:
    """Describe the `_set_intercept` positional payload in ElasticNet.fit."""
    return X_offset, y_offset, X_scale


def witness_cd_estimator_fit_return_self(estimator_identity: object) -> object:
    """Describe the final `return self` shell in ElasticNet.fit."""
    return estimator_identity
