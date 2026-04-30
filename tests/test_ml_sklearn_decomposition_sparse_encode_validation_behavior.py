from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.decomposition import sparse_encode as sklearn_sparse_encode


def test_sparse_encode_validation_imports() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_validation import (
        sparse_encode_require_matching_features,
        sparse_encode_require_positive_compatible_algorithm,
    )

    assert callable(sparse_encode_require_matching_features)
    assert callable(sparse_encode_require_positive_compatible_algorithm)


def test_sparse_encode_matching_feature_guard_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_validation import sparse_encode_require_matching_features

    dictionary = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    X_ok = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    X_bad = np.array([[1.0, 2.0]], dtype=np.float64)

    assert sparse_encode_require_matching_features(
        X_ok.shape[1],
        dictionary_feature_count=dictionary.shape[1],
    ) == X_ok.shape[1]

    with pytest.raises(ValueError, match="Dictionary and X have different numbers of features:"):
        sparse_encode_require_matching_features(2, dictionary_feature_count=3)

    with pytest.raises(ValueError, match="Dictionary and X have different numbers of features:"):
        sklearn_sparse_encode(X_bad, dictionary)


def test_sparse_encode_positive_algorithm_guard_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_validation import sparse_encode_require_positive_compatible_algorithm

    assert sparse_encode_require_positive_compatible_algorithm("lasso_cd", positive=False) == "lasso_cd"
    assert sparse_encode_require_positive_compatible_algorithm("threshold", positive=True) == "threshold"

    with pytest.raises(ValueError, match="Positive constraint not supported for 'omp' coding method."):
        sparse_encode_require_positive_compatible_algorithm("omp", positive=True)

    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    dictionary = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Positive constraint not supported for 'omp' coding method."):
        sklearn_sparse_encode(X, dictionary, algorithm="omp", positive=True)


def test_sparse_encode_validation_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_validation import (
        sparse_encode_require_matching_features,
        sparse_encode_require_positive_compatible_algorithm,
    )

    with pytest.raises(ViolationError):
        sparse_encode_require_matching_features(0, dictionary_feature_count=2)

    with pytest.raises(ViolationError):
        sparse_encode_require_matching_features(2, dictionary_feature_count=True)

    with pytest.raises(ViolationError):
        sparse_encode_require_positive_compatible_algorithm("bad", positive=False)

    with pytest.raises(ViolationError):
        sparse_encode_require_positive_compatible_algorithm("lasso_cd", positive=1)  # type: ignore[arg-type]
