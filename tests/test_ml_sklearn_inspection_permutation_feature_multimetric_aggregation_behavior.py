from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_permutation_feature_multimetric_aggregation_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_feature_multimetric_aggregation import (
        permutation_importance_feature_metric_names,
        permutation_importance_feature_metric_score_dict,
    )

    assert callable(permutation_importance_feature_metric_names)
    assert callable(permutation_importance_feature_metric_score_dict)


def test_permutation_feature_multimetric_aggregation_matches_sklearn_numeric_path() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_feature_multimetric_aggregation import (
        permutation_importance_feature_metric_names,
        permutation_importance_feature_metric_score_dict,
    )

    score_dicts = (
        {"roc_auc": 0.81, "accuracy": 0.75},
        {"roc_auc": 0.79, "accuracy": 0.74},
        {"roc_auc": 0.82, "accuracy": 0.76},
    )

    actual = permutation_importance_feature_metric_score_dict(score_dicts)

    assert permutation_importance_feature_metric_names(score_dicts) == ("roc_auc", "accuracy")
    assert tuple(actual.keys()) == ("roc_auc", "accuracy")
    assert np.array_equal(actual["roc_auc"], np.asarray([0.81, 0.79, 0.82], dtype=np.float64))
    assert np.array_equal(actual["accuracy"], np.asarray([0.75, 0.74, 0.76], dtype=np.float64))


def test_permutation_feature_multimetric_aggregation_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_feature_multimetric_aggregation import (
        permutation_importance_feature_metric_names,
        permutation_importance_feature_metric_score_dict,
    )

    with pytest.raises(ViolationError):
        permutation_importance_feature_metric_names(())

    with pytest.raises(ViolationError):
        permutation_importance_feature_metric_score_dict(({"roc_auc": np.nan},))
