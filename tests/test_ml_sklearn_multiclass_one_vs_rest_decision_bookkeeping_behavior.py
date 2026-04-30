from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def test_one_vs_rest_decision_bookkeeping_imports() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_decision_bookkeeping import (
        one_vs_rest_decision_stack,
    )

    assert callable(one_vs_rest_decision_stack)


def test_one_vs_rest_decision_stack_matches_sklearn_inner_stack() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_decision_bookkeeping import (
        one_vs_rest_decision_stack,
    )

    X, y = load_iris(return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X, y)
    decision_blocks = tuple(np.asarray(estimator.decision_function(X).ravel(), dtype=np.float64) for estimator in ovr.estimators_)

    observed = one_vs_rest_decision_stack(decision_blocks)
    expected = np.asarray(decision_blocks, dtype=np.float64)

    assert np.allclose(observed, expected)


def test_one_vs_rest_decision_stack_feeds_existing_output_shaper() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_decision_bookkeeping import (
        one_vs_rest_decision_stack,
    )
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postprocessing import (
        one_vs_rest_decision_output,
    )

    decision_blocks = (
        np.array([0.1, 0.2], dtype=np.float64),
        np.array([-0.5, 0.7], dtype=np.float64),
        np.array([1.0, -1.0], dtype=np.float64),
    )

    stacked = one_vs_rest_decision_stack(decision_blocks)
    shaped = one_vs_rest_decision_output(stacked)

    assert stacked.shape == (3, 2)
    assert shaped.shape == (2, 3)
    assert np.allclose(shaped, np.asarray(decision_blocks, dtype=np.float64).T)


def test_one_vs_rest_decision_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_decision_bookkeeping import (
        one_vs_rest_decision_stack,
    )

    with pytest.raises(ViolationError):
        one_vs_rest_decision_stack(())

    with pytest.raises(ViolationError):
        one_vs_rest_decision_stack(
            (
                np.array([0.1, 0.2], dtype=np.float64),
                np.array([[0.3, 0.4]], dtype=np.float64),
            )
        )

    with pytest.raises(ViolationError):
        one_vs_rest_decision_stack(
            (
                np.array([0.1], dtype=np.float64),
                np.array([0.2, 0.3], dtype=np.float64),
            )
        )
