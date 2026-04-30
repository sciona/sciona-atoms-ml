"""t-SNE fit-transform bookkeeping helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tsne_fit_transform_max_iter,
    witness_tsne_fit_transform_require_single_iter_source,
    witness_tsne_n_features_out,
    witness_tsne_pairwise_input_tag,
)


def _optional_positive_int(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _iter_spec_valid(value: object) -> bool:
    return value == "deprecated" or (isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _metric_name_valid(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _embedding_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


@register_atom(witness_tsne_fit_transform_require_single_iter_source)
@icontract.require(lambda n_iter: _iter_spec_valid(n_iter), "n_iter must be 'deprecated' or a positive integer")
@icontract.require(lambda max_iter: _optional_positive_int(max_iter), "max_iter must be None or a positive integer")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def tsne_fit_transform_require_single_iter_source(
    n_iter: str | int,
    max_iter: int | None,
) -> bool:
    """Require sklearn's fit_transform rule that n_iter and max_iter are not both explicitly set."""
    if n_iter != "deprecated" and max_iter is not None:
        raise ValueError(
            "Both 'n_iter' and 'max_iter' attributes were set. Attribute"
            " 'n_iter' was deprecated in version 1.5 and will be removed in"
            " 1.7. To avoid this error, only set the 'max_iter' attribute."
        )
    return True


@register_atom(witness_tsne_fit_transform_max_iter)
@icontract.require(lambda n_iter: _iter_spec_valid(n_iter), "n_iter must be 'deprecated' or a positive integer")
@icontract.require(lambda max_iter: _optional_positive_int(max_iter), "max_iter must be None or a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "_max_iter must be a positive integer")
def tsne_fit_transform_max_iter(
    n_iter: str | int,
    max_iter: int | None,
) -> int:
    """Resolve sklearn's effective _max_iter value in TSNE.fit_transform."""
    tsne_fit_transform_require_single_iter_source(n_iter, max_iter)
    if n_iter != "deprecated":
        return int(n_iter)
    if max_iter is None:
        return 1000
    return int(max_iter)


@register_atom(witness_tsne_n_features_out)
@icontract.require(lambda embedding: _embedding_valid(embedding), "embedding must be a finite 2D matrix")
@icontract.ensure(lambda result: _positive_int(result), "n_features_out must be a positive integer")
def tsne_n_features_out(
    embedding: NDArray[np.float64],
) -> int:
    """Return sklearn's transformed output width from a fitted t-SNE embedding matrix."""
    return int(np.asarray(embedding, dtype=np.float64).shape[1])


@register_atom(witness_tsne_pairwise_input_tag)
@icontract.require(lambda metric: _metric_name_valid(metric), "metric must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def tsne_pairwise_input_tag(
    metric: str,
) -> bool:
    """Return sklearn's pairwise-input tag for t-SNE."""
    return metric == "precomputed"
