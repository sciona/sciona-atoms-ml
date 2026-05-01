"""HDBSCAN estimator-tag helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import witness_hdbscan_allow_nan_tag, witness_hdbscan_sparse_input_tag


@register_atom(witness_hdbscan_sparse_input_tag)
@icontract.require(lambda metric: isinstance(metric, str), "metric must be a string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def hdbscan_sparse_input_tag(metric: str) -> bool:
    """Return HDBSCAN's sparse-input estimator tag."""
    del metric
    return True


@register_atom(witness_hdbscan_allow_nan_tag)
@icontract.require(lambda metric: isinstance(metric, str), "metric must be a string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def hdbscan_allow_nan_tag(metric: str) -> bool:
    """Return HDBSCAN's allow-nan estimator tag from the metric choice."""
    return metric != "precomputed"
