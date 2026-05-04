from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_permutation_output_branching_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_output_branching_shell import (
        permutation_importance_metric_names,
        permutation_importance_single_metric_score_matrix,
        permutation_importance_use_multimetric_results,
    )

    assert callable(permutation_importance_use_multimetric_results)
    assert callable(permutation_importance_metric_names)
    assert callable(permutation_importance_single_metric_score_matrix)


def test_permutation_output_branching_shell_matches_sklearn_tail_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_output_branching_shell import (
        permutation_importance_metric_names,
        permutation_importance_single_metric_score_matrix,
        permutation_importance_use_multimetric_results,
    )

    baseline_score = 0.875
    baseline_scores = {"roc_auc": 0.91, "accuracy": 0.83}
    scores = [[0.80, 0.79, 0.81], [0.50, 0.51, 0.49]]

    assert permutation_importance_use_multimetric_results(baseline_score) is False
    assert permutation_importance_use_multimetric_results(baseline_scores) is True
    assert permutation_importance_metric_names(baseline_scores) == ("roc_auc", "accuracy")
    assert np.array_equal(
        permutation_importance_single_metric_score_matrix(scores),
        np.asarray(scores, dtype=np.float64),
    )


def test_permutation_output_branching_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_output_branching_shell import (
        permutation_importance_metric_names,
        permutation_importance_single_metric_score_matrix,
        permutation_importance_use_multimetric_results,
    )

    with pytest.raises(ViolationError):
        permutation_importance_use_multimetric_results({})

    with pytest.raises(ViolationError):
        permutation_importance_metric_names({"roc_auc": np.nan})

    with pytest.raises(ViolationError):
        permutation_importance_single_metric_score_matrix([1.0, 2.0])
