from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_best_update_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_best_update_shell import (
        cd_cv_best_candidate_count,
        cd_cv_best_candidate_triples,
        cd_cv_best_mse_improved,
        cd_cv_fit_best_alpha,
        cd_cv_fit_best_l1_ratio,
    )

    assert callable(cd_cv_best_candidate_triples)
    assert callable(cd_cv_best_candidate_count)
    assert callable(cd_cv_best_mse_improved)
    assert callable(cd_cv_fit_best_l1_ratio)
    assert callable(cd_cv_fit_best_alpha)


def test_coordinate_descent_cv_best_update_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_best_update_shell import (
        cd_cv_best_candidate_count,
        cd_cv_best_candidate_triples,
        cd_cv_best_mse_improved,
        cd_cv_fit_best_alpha,
        cd_cv_fit_best_l1_ratio,
    )

    l1_ratios = [0.2, 0.8]
    alphas = [np.array([2.0, 1.0]), np.array([1.5, 0.5])]
    mean_mse = [np.array([0.4, 0.3]), np.array([0.5, 0.2])]
    triples = cd_cv_best_candidate_triples(l1_ratios, alphas, mean_mse)
    assert len(triples) == 2
    assert triples[0][0] == 0.2
    assert np.array_equal(triples[0][1], np.array([2.0, 1.0]))
    assert np.array_equal(triples[0][2], np.array([0.4, 0.3]))
    assert cd_cv_best_candidate_count(triples) == 2
    assert cd_cv_best_mse_improved(float(np.inf), 0.3) is True
    assert cd_cv_best_mse_improved(0.3, 0.3) is False
    assert cd_cv_fit_best_l1_ratio(0.8) == 0.8
    assert np.isclose(cd_cv_fit_best_alpha(0.5), 0.5)


def test_coordinate_descent_cv_best_update_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_best_update_shell import (
        cd_cv_best_candidate_triples,
        cd_cv_best_mse_improved,
        cd_cv_fit_best_alpha,
    )

    with pytest.raises(ViolationError):
        cd_cv_best_candidate_triples([0.2], [np.array([1.0])], [])

    with pytest.raises(ViolationError):
        cd_cv_best_mse_improved(-np.inf, 0.2)

    with pytest.raises(ViolationError):
        cd_cv_fit_best_alpha(np.nan)
