"""Ghost witnesses for sklearn multioutput chain prediction bookkeeping helpers."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from sciona.ghost.abstract import AbstractArray


def witness_chain_prediction_method_name(chain_method_name: str | None = None) -> str:
    """Describe sklearn's chain prediction method-name fallback."""
    if chain_method_name is None:
        return "predict"
    if not isinstance(chain_method_name, str) or len(chain_method_name) < 1:
        raise ValueError("chain_method_name must be None or a nonempty string")
    return chain_method_name


def witness_chain_prediction_output_buffer(n_samples: int, n_outputs: int) -> AbstractArray:
    """Describe the zero-initialized chain output-prediction buffer."""
    if n_samples < 1 or n_outputs < 1:
        raise ValueError("n_samples and n_outputs must be positive")
    return AbstractArray(shape=(n_samples, n_outputs), dtype="float64")


def witness_chain_prediction_feature_buffer(n_samples: int, n_outputs: int) -> AbstractArray:
    """Describe the zero-initialized chain feature-prediction buffer."""
    if n_samples < 1 or n_outputs < 1:
        raise ValueError("n_samples and n_outputs must be positive")
    return AbstractArray(shape=(n_samples, n_outputs), dtype="float64")


def witness_chain_prediction_previous_predictions(
    feature_chain: AbstractArray,
    chain_idx: int,
) -> AbstractArray:
    """Describe the prefix of already-filled prediction columns for one chain step."""
    if len(feature_chain.shape) != 2:
        raise ValueError("feature_chain must be 2D")
    n_samples = int(feature_chain.shape[0])
    n_outputs = int(feature_chain.shape[1])
    if n_samples < 1 or n_outputs < 1 or chain_idx < 0 or chain_idx > n_outputs:
        raise ValueError("feature_chain must be nonempty and chain_idx must select a valid prefix")
    return AbstractArray(shape=(n_samples, chain_idx), dtype="float64")


def witness_chain_sparse_hstack_base(X: AbstractArray) -> csr_matrix:
    """Describe prediction-time sparse input normalization before hstack."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    if n_samples < 1 or n_features < 1:
        raise ValueError("X must be nonempty")
    return csr_matrix((n_samples, n_features), dtype=float)
