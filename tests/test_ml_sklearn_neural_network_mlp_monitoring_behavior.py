from __future__ import annotations

import numpy as np
import pytest


def test_mlp_monitoring_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_monitoring import (
        mlp_monitor_best_loss,
        mlp_monitor_best_validation_score,
        mlp_monitor_defaults,
        mlp_monitor_loss_no_improvement_count,
        mlp_monitor_validation_no_improvement_count,
    )

    assert callable(mlp_monitor_defaults)
    assert callable(mlp_monitor_best_loss)
    assert callable(mlp_monitor_loss_no_improvement_count)
    assert callable(mlp_monitor_best_validation_score)
    assert callable(mlp_monitor_validation_no_improvement_count)


def test_mlp_monitor_defaults_match_source_stochastic_initialization() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_monitoring import mlp_monitor_defaults

    assert mlp_monitor_defaults(early_stopping=False) == (float("inf"), None, 0, None)
    assert mlp_monitor_defaults(early_stopping=True) == (None, float("-inf"), 0, ())


def test_mlp_monitor_loss_branch_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_monitoring import (
        mlp_monitor_best_loss,
        mlp_monitor_loss_no_improvement_count,
    )

    assert mlp_monitor_loss_no_improvement_count(0.8, 1.0, tol=1e-4, no_improvement_count=3) == 0
    assert mlp_monitor_best_loss(0.8, 1.0) == 0.8

    assert mlp_monitor_loss_no_improvement_count(1.0, 1.0, tol=1e-4, no_improvement_count=3) == 4
    assert mlp_monitor_best_loss(1.0, 0.8) == 0.8


def test_mlp_monitor_validation_branch_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_monitoring import (
        mlp_monitor_best_validation_score,
        mlp_monitor_validation_no_improvement_count,
    )

    assert mlp_monitor_validation_no_improvement_count(
        0.7,
        0.6,
        tol=1e-4,
        no_improvement_count=2,
    ) == 0
    assert mlp_monitor_best_validation_score(0.7, 0.6) == 0.7

    assert mlp_monitor_validation_no_improvement_count(
        0.6,
        0.6,
        tol=1e-4,
        no_improvement_count=2,
    ) == 3
    assert mlp_monitor_best_validation_score(0.6, 0.7) == 0.7


def test_mlp_monitor_validation_defaults_integrate_with_first_update() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_monitoring import (
        mlp_monitor_best_validation_score,
        mlp_monitor_defaults,
        mlp_monitor_validation_no_improvement_count,
    )

    best_loss, best_validation_score, no_improvement_count, validation_scores = mlp_monitor_defaults(
        early_stopping=True
    )
    assert best_loss is None
    assert validation_scores == ()
    assert mlp_monitor_validation_no_improvement_count(
        0.25,
        best_validation_score,
        tol=1e-4,
        no_improvement_count=no_improvement_count,
    ) == 0
    assert mlp_monitor_best_validation_score(0.25, best_validation_score) == 0.25


def test_mlp_monitoring_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_monitoring import (
        mlp_monitor_best_loss,
        mlp_monitor_best_validation_score,
        mlp_monitor_defaults,
        mlp_monitor_loss_no_improvement_count,
        mlp_monitor_validation_no_improvement_count,
    )

    with pytest.raises(Exception):
        mlp_monitor_defaults(early_stopping=np.bool_(True))
    with pytest.raises(Exception):
        mlp_monitor_best_loss(np.nan, 1.0)
    with pytest.raises(Exception):
        mlp_monitor_loss_no_improvement_count(1.0, 1.0, tol=-1.0, no_improvement_count=0)
    with pytest.raises(Exception):
        mlp_monitor_best_validation_score(0.5, np.nan)
    with pytest.raises(Exception):
        mlp_monitor_validation_no_improvement_count(0.5, 0.4, tol=0.0, no_improvement_count=np.int64(1))
