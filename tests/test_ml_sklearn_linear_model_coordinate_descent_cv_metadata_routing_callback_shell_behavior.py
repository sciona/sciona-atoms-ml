from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_metadata_routing_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_routing_callback_shell import (
        cd_cv_process_routing_args,
        cd_cv_process_routing_kwargs,
        cd_cv_routed_params_result,
        cd_cv_routing_params_with_sample_weight,
        cd_cv_splitter_consumes_kwargs,
        cd_cv_splitter_supports_sample_weight_result,
    )

    assert callable(cd_cv_splitter_consumes_kwargs)
    assert callable(cd_cv_splitter_supports_sample_weight_result)
    assert callable(cd_cv_routing_params_with_sample_weight)
    assert callable(cd_cv_process_routing_args)
    assert callable(cd_cv_process_routing_kwargs)
    assert callable(cd_cv_routed_params_result)


def test_coordinate_descent_cv_metadata_routing_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_routing_callback_shell import (
        cd_cv_process_routing_args,
        cd_cv_process_routing_kwargs,
        cd_cv_routed_params_result,
        cd_cv_routing_params_with_sample_weight,
        cd_cv_splitter_consumes_kwargs,
        cd_cv_splitter_supports_sample_weight_result,
    )

    cv = object()
    assert cd_cv_splitter_consumes_kwargs(cv) == {
        "method": "split",
        "params": ["sample_weight"],
    }
    assert cd_cv_splitter_supports_sample_weight_result(True) is True

    params = {"groups": [0, 1, 0]}
    sample_weight = [1.0, 2.0, 3.0]
    forwarded = cd_cv_routing_params_with_sample_weight(params, True, sample_weight)
    assert forwarded == {"groups": [0, 1, 0], "sample_weight": sample_weight}
    assert forwarded["sample_weight"] is sample_weight
    assert cd_cv_routing_params_with_sample_weight(params, False, sample_weight) == params

    estimator = object()
    assert cd_cv_process_routing_args(estimator) == (estimator, "fit")
    assert cd_cv_process_routing_args(estimator)[0] is estimator
    assert cd_cv_process_routing_kwargs(forwarded) == forwarded

    routed_params = object()
    assert cd_cv_routed_params_result(routed_params) is routed_params


def test_coordinate_descent_cv_metadata_routing_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_routing_callback_shell import (
        cd_cv_process_routing_kwargs,
        cd_cv_routing_params_with_sample_weight,
        cd_cv_splitter_supports_sample_weight_result,
    )

    with pytest.raises(ViolationError):
        cd_cv_splitter_supports_sample_weight_result("yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_routing_params_with_sample_weight([], True, [1.0])  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_process_routing_kwargs([("groups", [0])])  # type: ignore[arg-type]
