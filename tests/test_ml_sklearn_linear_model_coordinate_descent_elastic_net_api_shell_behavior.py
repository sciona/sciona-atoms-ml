from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_elastic_net_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_api_shell import (
        cd_elastic_net_init_attributes,
        cd_elastic_net_path_name,
        cd_elastic_net_sparse_decision_output,
        cd_elastic_net_sparse_decision_required,
        cd_elastic_net_sparse_dot_args,
        cd_elastic_net_sparse_dot_kwargs,
        cd_elastic_net_sparse_input_tag,
    )

    assert callable(cd_elastic_net_path_name)
    assert callable(cd_elastic_net_init_attributes)
    assert callable(cd_elastic_net_sparse_decision_required)
    assert callable(cd_elastic_net_sparse_dot_args)
    assert callable(cd_elastic_net_sparse_dot_kwargs)
    assert callable(cd_elastic_net_sparse_decision_output)
    assert callable(cd_elastic_net_sparse_input_tag)


def test_coordinate_descent_elastic_net_api_shell_matches_sklearn_api() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_api_shell import (
        cd_elastic_net_init_attributes,
        cd_elastic_net_path_name,
        cd_elastic_net_sparse_decision_output,
        cd_elastic_net_sparse_decision_required,
        cd_elastic_net_sparse_dot_args,
        cd_elastic_net_sparse_dot_kwargs,
        cd_elastic_net_sparse_input_tag,
    )

    assert cd_elastic_net_path_name("elastic_net") == "enet_path"

    alpha = object()
    l1_ratio = object()
    precompute = object()
    tol = object()
    random_state = object()
    attrs = cd_elastic_net_init_attributes(
        alpha,
        l1_ratio,
        True,
        precompute,
        1000,
        False,
        tol,
        True,
        False,
        random_state,
        "random",
    )
    assert attrs == {
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "fit_intercept": True,
        "precompute": precompute,
        "max_iter": 1000,
        "copy_X": False,
        "tol": tol,
        "warm_start": True,
        "positive": False,
        "random_state": random_state,
        "selection": "random",
    }
    assert attrs["alpha"] is alpha
    assert attrs["l1_ratio"] is l1_ratio
    assert attrs["precompute"] is precompute
    assert attrs["tol"] is tol
    assert attrs["random_state"] is random_state

    assert cd_elastic_net_sparse_decision_required(True) is True
    assert cd_elastic_net_sparse_decision_required(False) is False

    X = object()
    coef = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dot_args = cd_elastic_net_sparse_dot_args(X, coef)
    assert dot_args[0] is X
    assert np.array_equal(dot_args[1], coef.T)
    assert cd_elastic_net_sparse_dot_kwargs(True) == {"dense_output": True}

    dot_output = np.array([[1.0, 2.0], [3.0, 4.0]])
    intercept = np.array([0.5, -0.25])
    assert np.array_equal(
        cd_elastic_net_sparse_decision_output(dot_output, intercept),
        dot_output + intercept,
    )
    assert cd_elastic_net_sparse_input_tag(False) is True


def test_coordinate_descent_elastic_net_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_elastic_net_api_shell import (
        cd_elastic_net_init_attributes,
        cd_elastic_net_path_name,
        cd_elastic_net_sparse_decision_required,
        cd_elastic_net_sparse_dot_args,
        cd_elastic_net_sparse_dot_kwargs,
    )

    with pytest.raises(ViolationError):
        cd_elastic_net_path_name("lasso")

    with pytest.raises(ViolationError):
        cd_elastic_net_init_attributes(
            1.0,
            0.5,
            True,
            False,
            0,
            True,
            1e-4,
            False,
            False,
            None,
            "cyclic",
        )

    with pytest.raises(ViolationError):
        cd_elastic_net_sparse_decision_required(1)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_elastic_net_sparse_dot_args(object(), np.array([[np.nan]]))

    with pytest.raises(ViolationError):
        cd_elastic_net_sparse_dot_kwargs(False)
