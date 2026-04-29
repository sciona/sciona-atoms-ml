from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_rfe_state_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping import (
        rfe_elimination_threshold,
        rfe_final_feature_count,
        rfe_initial_ranking,
        rfe_initial_step_history,
        rfe_initial_support_mask,
    )

    assert callable(rfe_initial_support_mask)
    assert callable(rfe_initial_ranking)
    assert callable(rfe_initial_step_history)
    assert callable(rfe_elimination_threshold)
    assert callable(rfe_final_feature_count)


def test_rfe_initial_support_mask_matches_sklearn_initial_state() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping import rfe_initial_support_mask

    observed = rfe_initial_support_mask(4)

    assert np.array_equal(observed, np.array([True, True, True, True], dtype=np.bool_))


def test_rfe_initial_ranking_matches_sklearn_initial_state() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping import rfe_initial_ranking

    observed = rfe_initial_ranking(4)

    assert np.array_equal(observed, np.array([1, 1, 1, 1], dtype=np.int64))


def test_rfe_initial_step_history_is_empty() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping import rfe_initial_step_history

    step_n_features, step_scores = rfe_initial_step_history()

    assert step_n_features.shape == (0,)
    assert step_scores.shape == (0,)


def test_rfe_elimination_threshold_matches_sklearn_min_logic() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping import rfe_elimination_threshold

    assert rfe_elimination_threshold(6, n_features_to_select=2, step=3) == 3
    assert rfe_elimination_threshold(6, n_features_to_select=4, step=3) == 2


def test_rfe_final_feature_count_matches_support_sum() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping import rfe_final_feature_count

    support_mask = np.array([True, False, True, False, True], dtype=np.bool_)

    assert rfe_final_feature_count(support_mask) == 3


def test_rfe_state_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping import (
        rfe_elimination_threshold,
        rfe_initial_support_mask,
    )

    with pytest.raises(ViolationError):
        rfe_initial_support_mask(0)

    with pytest.raises(ViolationError):
        rfe_elimination_threshold(3, n_features_to_select=3, step=1)
