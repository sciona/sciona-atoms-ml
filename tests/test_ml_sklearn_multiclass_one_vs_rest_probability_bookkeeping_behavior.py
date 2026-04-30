from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def test_one_vs_rest_probability_bookkeeping_imports() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_probability_bookkeeping import (
        one_vs_rest_positive_probability_stack,
    )

    assert callable(one_vs_rest_positive_probability_stack)


def test_positive_probability_stack_matches_one_vs_rest_predict_proba_inner_stack() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_probability_bookkeeping import (
        one_vs_rest_positive_probability_stack,
    )

    X, y = load_iris(return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X, y)
    probability_blocks = tuple(np.asarray(estimator.predict_proba(X), dtype=np.float64) for estimator in ovr.estimators_)

    observed = one_vs_rest_positive_probability_stack(probability_blocks)
    expected = np.asarray([block[:, 1] for block in probability_blocks], dtype=np.float64)

    assert np.allclose(observed, expected)


def test_positive_probability_stack_feeds_existing_postprocessing_helpers() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_probability_bookkeeping import (
        one_vs_rest_positive_probability_stack,
    )
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postprocessing import (
        one_vs_rest_normalized_probability_matrix,
        one_vs_rest_positive_probability_matrix,
    )

    probability_blocks = (
        np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float64),
        np.array([[0.4, 0.6], [0.6, 0.4]], dtype=np.float64),
        np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64),
    )

    stacked = one_vs_rest_positive_probability_stack(probability_blocks)
    matrix = one_vs_rest_positive_probability_matrix(stacked)
    normalized = one_vs_rest_normalized_probability_matrix(matrix)

    assert stacked.shape == (3, 2)
    assert matrix.shape == (2, 3)
    assert np.allclose(np.sum(normalized, axis=1), 1.0)


def test_one_vs_rest_probability_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_probability_bookkeeping import (
        one_vs_rest_positive_probability_stack,
    )

    with pytest.raises(ViolationError):
        one_vs_rest_positive_probability_stack(())

    with pytest.raises(ViolationError):
        one_vs_rest_positive_probability_stack(
            (
                np.array([[1.0]], dtype=np.float64),
            )
        )
