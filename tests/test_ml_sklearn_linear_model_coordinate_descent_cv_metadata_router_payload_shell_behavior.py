from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_metadata_router_payload_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_payload_shell import (
        cd_cv_metadata_router_result,
        cd_cv_metadata_router_self_request,
        cd_cv_metadata_router_splitter_payload,
    )

    assert callable(cd_cv_metadata_router_self_request)
    assert callable(cd_cv_metadata_router_splitter_payload)
    assert callable(cd_cv_metadata_router_result)


def test_coordinate_descent_cv_metadata_router_payload_shell_preserves_identities() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_payload_shell import (
        cd_cv_metadata_router_result,
        cd_cv_metadata_router_self_request,
        cd_cv_metadata_router_splitter_payload,
    )

    estimator = object()
    splitter = object()
    method_mapping = {"caller": "fit", "callee": "split"}
    router = object()

    payload = cd_cv_metadata_router_splitter_payload(splitter, method_mapping)

    assert cd_cv_metadata_router_self_request(estimator) is estimator
    assert payload == {"splitter": splitter, "method_mapping": method_mapping}
    assert payload["splitter"] is splitter
    assert payload["method_mapping"] is method_mapping
    assert cd_cv_metadata_router_result(router) is router


def test_coordinate_descent_cv_metadata_router_payload_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_metadata_router_payload_shell import (
        cd_cv_metadata_router_splitter_payload,
    )

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_splitter_payload(object(), {"caller": "fit", "callee": "score"})

    with pytest.raises(ViolationError):
        cd_cv_metadata_router_splitter_payload(object(), {"caller": "predict", "callee": "split"})
