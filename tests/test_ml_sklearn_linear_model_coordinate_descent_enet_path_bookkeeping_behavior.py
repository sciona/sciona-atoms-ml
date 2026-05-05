from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_bookkeeping import (
        cd_enet_path_multi_output,
        cd_enet_path_outputs,
        cd_enet_path_positive_multi_output_guard_required,
        cd_enet_path_random_selection,
        cd_enet_path_regularization_pair,
        cd_enet_path_sorted_alphas,
        cd_enet_path_target_count,
    )

    assert callable(cd_enet_path_multi_output)
    assert callable(cd_enet_path_target_count)
    assert callable(cd_enet_path_positive_multi_output_guard_required)
    assert callable(cd_enet_path_sorted_alphas)
    assert callable(cd_enet_path_random_selection)
    assert callable(cd_enet_path_regularization_pair)
    assert callable(cd_enet_path_outputs)


def test_coordinate_descent_enet_path_bookkeeping_matches_sklearn_formulas() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_bookkeeping import (
        cd_enet_path_multi_output,
        cd_enet_path_outputs,
        cd_enet_path_positive_multi_output_guard_required,
        cd_enet_path_random_selection,
        cd_enet_path_regularization_pair,
        cd_enet_path_sorted_alphas,
        cd_enet_path_target_count,
    )

    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert cd_enet_path_multi_output(y.ndim) is True
    assert cd_enet_path_target_count(y.shape, True) == 2
    assert cd_enet_path_positive_multi_output_guard_required(True, True) is True
    assert np.array_equal(
        cd_enet_path_sorted_alphas(np.array([0.1, 2.0, 0.5], dtype=np.float64)),
        np.array([2.0, 0.5, 0.1], dtype=np.float64),
    )
    assert cd_enet_path_random_selection("random") is True
    assert cd_enet_path_regularization_pair(alpha=0.5, l1_ratio=0.2, n_samples=10) == (1.0, 4.0)

    outputs = cd_enet_path_outputs(
        np.array([2.0, 1.0], dtype=np.float64),
        np.zeros((3, 2), dtype=np.float64),
        np.array([0.1, 0.2], dtype=np.float64),
        [5, 6],
        True,
    )
    assert len(outputs) == 4
    assert outputs[3] == [5, 6]


def test_coordinate_descent_enet_path_bookkeeping_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_bookkeeping import (
        cd_enet_path_multi_output,
        cd_enet_path_random_selection,
        cd_enet_path_sorted_alphas,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_multi_output(0)

    with pytest.raises(ViolationError):
        cd_enet_path_random_selection("bad")

    with pytest.raises(ViolationError):
        cd_enet_path_sorted_alphas(np.array([], dtype=np.float64))
