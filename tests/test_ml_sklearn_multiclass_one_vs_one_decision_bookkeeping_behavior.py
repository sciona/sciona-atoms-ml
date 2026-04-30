from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_one_vs_one_decision_bookkeeping_imports() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_one_decision_bookkeeping import (
        one_vs_one_decision_feature_blocks,
        one_vs_one_decision_output,
    )

    assert callable(one_vs_one_decision_feature_blocks)
    assert callable(one_vs_one_decision_output)


def test_one_vs_one_decision_feature_blocks_repeat_full_X_when_pairwise_indices_missing() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_one_decision_bookkeeping import (
        one_vs_one_decision_feature_blocks,
    )

    X = np.arange(12, dtype=np.float64).reshape(3, 4)
    blocks = one_vs_one_decision_feature_blocks(X, estimator_count=3)

    assert len(blocks) == 3
    for block in blocks:
        assert np.array_equal(block, X)


def test_one_vs_one_decision_feature_blocks_match_pairwise_column_slices() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_one_decision_bookkeeping import (
        one_vs_one_decision_feature_blocks,
    )

    X = np.arange(20, dtype=np.float64).reshape(5, 4)
    pairwise_indices = ((0, 2), (1, 3))
    blocks = one_vs_one_decision_feature_blocks(
        X,
        estimator_count=2,
        pairwise_indices=pairwise_indices,
    )

    assert len(blocks) == 2
    assert np.array_equal(blocks[0], X[:, pairwise_indices[0]])
    assert np.array_equal(blocks[1], X[:, pairwise_indices[1]])


def test_one_vs_one_decision_output_matches_binary_squeeze_and_multiclass_passthrough() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_one_decision_bookkeeping import (
        one_vs_one_decision_output,
    )

    binary_scores = np.array(
        [[-0.5, 0.5], [0.25, 1.25], [2.0, 3.0]],
        dtype=np.float64,
    )
    multiclass_scores = np.array(
        [[1.0, 0.5, -1.0], [0.1, 0.2, 0.3]],
        dtype=np.float64,
    )

    assert np.array_equal(
        one_vs_one_decision_output(binary_scores, n_classes=2),
        binary_scores[:, 1],
    )
    assert np.array_equal(
        one_vs_one_decision_output(multiclass_scores, n_classes=3),
        multiclass_scores,
    )


def test_one_vs_one_decision_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_one_decision_bookkeeping import (
        one_vs_one_decision_feature_blocks,
        one_vs_one_decision_output,
    )

    X = np.arange(6, dtype=np.float64).reshape(3, 2)

    with pytest.raises(ViolationError):
        one_vs_one_decision_feature_blocks(X, estimator_count=0)

    with pytest.raises(ViolationError):
        one_vs_one_decision_feature_blocks(
            X,
            estimator_count=2,
            pairwise_indices=((0,),),
        )

    with pytest.raises(ViolationError):
        one_vs_one_decision_output(np.array([[1.0, 2.0]], dtype=np.float64), n_classes=3)
