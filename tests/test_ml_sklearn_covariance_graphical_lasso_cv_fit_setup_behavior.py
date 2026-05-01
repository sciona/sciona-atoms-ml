from __future__ import annotations

import numpy as np
import pytest


def test_graphical_lasso_cv_fit_setup_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_fit_setup import (
        graphical_lasso_cv_explicit_alphas,
        graphical_lasso_cv_inner_verbose,
        graphical_lasso_cv_location,
        graphical_lasso_cv_refinement_count,
        graphical_lasso_cv_use_explicit_alphas,
    )

    assert callable(graphical_lasso_cv_location)
    assert callable(graphical_lasso_cv_inner_verbose)
    assert callable(graphical_lasso_cv_use_explicit_alphas)
    assert callable(graphical_lasso_cv_explicit_alphas)
    assert callable(graphical_lasso_cv_refinement_count)


def test_graphical_lasso_cv_location_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_fit_setup import (
        graphical_lasso_cv_location,
    )

    X = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float64)
    assert np.array_equal(graphical_lasso_cv_location(X, True), np.zeros(2))
    assert np.allclose(graphical_lasso_cv_location(X, False), X.mean(axis=0))


def test_graphical_lasso_cv_inner_verbose_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_fit_setup import (
        graphical_lasso_cv_inner_verbose,
    )

    assert graphical_lasso_cv_inner_verbose(0) == 0
    assert graphical_lasso_cv_inner_verbose(1) == 0
    assert graphical_lasso_cv_inner_verbose(3) == 2


def test_graphical_lasso_cv_alpha_mode_helpers_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_fit_setup import (
        graphical_lasso_cv_explicit_alphas,
        graphical_lasso_cv_refinement_count,
        graphical_lasso_cv_use_explicit_alphas,
    )

    explicit = [0.3, 0.1, 0.0]
    assert graphical_lasso_cv_use_explicit_alphas(explicit) is True
    assert np.array_equal(graphical_lasso_cv_explicit_alphas(explicit), np.array(explicit, dtype=np.float64))
    assert graphical_lasso_cv_refinement_count(True, 4) == 1

    assert graphical_lasso_cv_use_explicit_alphas(6) is False
    assert graphical_lasso_cv_refinement_count(False, 4) == 4


def test_graphical_lasso_cv_fit_setup_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_fit_setup import (
        graphical_lasso_cv_explicit_alphas,
        graphical_lasso_cv_inner_verbose,
        graphical_lasso_cv_location,
        graphical_lasso_cv_refinement_count,
    )

    with pytest.raises(Exception):
        graphical_lasso_cv_location(np.array([[1.0]]), False)
    with pytest.raises(Exception):
        graphical_lasso_cv_inner_verbose(True)
    with pytest.raises(Exception):
        graphical_lasso_cv_explicit_alphas([0.1, -0.1])
    with pytest.raises(Exception):
        graphical_lasso_cv_refinement_count(False, 0)
