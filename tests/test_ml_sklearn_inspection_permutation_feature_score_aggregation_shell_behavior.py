from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_permutation_feature_score_aggregation_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_feature_score_aggregation_shell import (
        permutation_importance_feature_scores_are_multimetric,
        permutation_importance_single_feature_score_vector,
    )

    assert callable(permutation_importance_feature_scores_are_multimetric)
    assert callable(permutation_importance_single_feature_score_vector)


def test_permutation_feature_score_aggregation_shell_matches_sklearn_tail_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_feature_score_aggregation_shell import (
        permutation_importance_feature_scores_are_multimetric,
        permutation_importance_single_feature_score_vector,
    )

    first_scalar_score = 0.81
    first_metric_score = {"roc_auc": 0.81, "accuracy": 0.75}
    scores = [0.81, 0.79, 0.82]

    assert permutation_importance_feature_scores_are_multimetric(first_scalar_score) is False
    assert permutation_importance_feature_scores_are_multimetric(first_metric_score) is True
    assert np.array_equal(
        permutation_importance_single_feature_score_vector(scores),
        np.asarray(scores, dtype=np.float64),
    )


def test_permutation_feature_score_aggregation_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_feature_score_aggregation_shell import (
        permutation_importance_feature_scores_are_multimetric,
        permutation_importance_single_feature_score_vector,
    )

    with pytest.raises(ViolationError):
        permutation_importance_feature_scores_are_multimetric({})

    with pytest.raises(ViolationError):
        permutation_importance_single_feature_score_vector([[1.0, 2.0]])
