from __future__ import annotations

import pytest


def test_mlp_stochastic_epoch_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_stochastic_epoch import (
        mlp_epoch_loss,
        mlp_restore_best_parameters_required,
        mlp_stochastic_incremental_break_required,
        mlp_stochastic_max_iter_warning_required,
        mlp_stochastic_no_improvement_count_after_trigger,
        mlp_stochastic_stop_message,
        mlp_time_step,
    )

    assert callable(mlp_epoch_loss)
    assert callable(mlp_time_step)
    assert callable(mlp_stochastic_stop_message)
    assert callable(mlp_stochastic_no_improvement_count_after_trigger)
    assert callable(mlp_stochastic_incremental_break_required)
    assert callable(mlp_stochastic_max_iter_warning_required)
    assert callable(mlp_restore_best_parameters_required)


def test_mlp_epoch_loss_and_time_step_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_stochastic_epoch import (
        mlp_epoch_loss,
        mlp_time_step,
    )

    assert mlp_epoch_loss(9.0, n_training_samples=3) == 3.0
    assert mlp_time_step(0, n_training_samples=8) == 8
    assert mlp_time_step(8, n_training_samples=8) == 16


def test_mlp_stochastic_stop_messages_match_source_formatting() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_stochastic_epoch import (
        mlp_stochastic_stop_message,
    )

    assert (
        mlp_stochastic_stop_message(early_stopping=True, tol=1e-4, n_iter_no_change=10)
        == "Validation score did not improve more than tol=0.000100 for 10 consecutive epochs."
    )
    assert (
        mlp_stochastic_stop_message(early_stopping=False, tol=1e-4, n_iter_no_change=10)
        == "Training loss did not improve more than tol=0.000100 for 10 consecutive epochs."
    )


def test_mlp_stochastic_post_trigger_and_break_flags_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_stochastic_epoch import (
        mlp_restore_best_parameters_required,
        mlp_stochastic_incremental_break_required,
        mlp_stochastic_no_improvement_count_after_trigger,
    )

    assert mlp_stochastic_no_improvement_count_after_trigger(is_stopping=True, no_improvement_count=4) == 4
    assert mlp_stochastic_no_improvement_count_after_trigger(is_stopping=False, no_improvement_count=4) == 0
    assert mlp_stochastic_incremental_break_required(incremental=True) is True
    assert mlp_stochastic_incremental_break_required(incremental=False) is False
    assert mlp_restore_best_parameters_required(early_stopping=True) is True
    assert mlp_restore_best_parameters_required(early_stopping=False) is False


def test_mlp_stochastic_max_iter_warning_guard_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_stochastic_epoch import (
        mlp_stochastic_max_iter_warning_required,
    )

    assert mlp_stochastic_max_iter_warning_required(200, max_iter=200, incremental=False) is True
    assert mlp_stochastic_max_iter_warning_required(199, max_iter=200, incremental=False) is False
    assert mlp_stochastic_max_iter_warning_required(200, max_iter=200, incremental=True) is False


def test_mlp_stochastic_epoch_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.neural_network.mlp_stochastic_epoch import (
        mlp_epoch_loss,
        mlp_restore_best_parameters_required,
        mlp_stochastic_incremental_break_required,
        mlp_stochastic_max_iter_warning_required,
        mlp_stochastic_no_improvement_count_after_trigger,
        mlp_stochastic_stop_message,
        mlp_time_step,
    )

    with pytest.raises(Exception):
        mlp_epoch_loss(float("nan"), n_training_samples=3)
    with pytest.raises(Exception):
        mlp_time_step(-1, n_training_samples=3)
    with pytest.raises(Exception):
        mlp_stochastic_stop_message(early_stopping=True, tol=1e-4, n_iter_no_change=0)
    with pytest.raises(Exception):
        mlp_stochastic_no_improvement_count_after_trigger(is_stopping=False, no_improvement_count=-1)
    with pytest.raises(Exception):
        mlp_stochastic_incremental_break_required(incremental=1)
    with pytest.raises(Exception):
        mlp_stochastic_max_iter_warning_required(0, max_iter=1, incremental=False)
    with pytest.raises(Exception):
        mlp_restore_best_parameters_required(early_stopping=0)
