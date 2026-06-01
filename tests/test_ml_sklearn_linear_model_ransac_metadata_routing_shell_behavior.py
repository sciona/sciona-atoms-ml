from __future__ import annotations

import pytest
from icontract import ViolationError


def test_ransac_metadata_routing_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_metadata_routing_shell import (
        ransac_metadata_estimator_payload,
        ransac_metadata_method_mapping_add_kwargs,
        ransac_metadata_router_owner,
        ransac_metadata_router_result,
    )

    assert callable(ransac_metadata_router_owner)
    assert callable(ransac_metadata_method_mapping_add_kwargs)
    assert callable(ransac_metadata_estimator_payload)
    assert callable(ransac_metadata_router_result)


def test_ransac_metadata_router_owner_matches_class_name_payload() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_metadata_routing_shell import ransac_metadata_router_owner

    assert ransac_metadata_router_owner("RANSACRegressor") == "RANSACRegressor"


def test_ransac_metadata_method_mapping_add_kwargs_match_source_chain() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_metadata_routing_shell import ransac_metadata_method_mapping_add_kwargs

    pairs = (
        ("fit", "fit"),
        ("fit", "score"),
        ("score", "score"),
        ("predict", "predict"),
    )

    assert pairs == (
        ("fit", "fit"),
        ("fit", "score"),
        ("score", "score"),
        ("predict", "predict"),
    )
    assert [ransac_metadata_method_mapping_add_kwargs(caller, callee) for caller, callee in pairs] == [
        {"caller": "fit", "callee": "fit"},
        {"caller": "fit", "callee": "score"},
        {"caller": "score", "callee": "score"},
        {"caller": "predict", "callee": "predict"},
    ]


def test_ransac_metadata_estimator_payload_preserves_identities() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_metadata_routing_shell import ransac_metadata_estimator_payload

    estimator = object()
    method_mapping = (
        ("fit", "fit"),
        ("fit", "score"),
        ("score", "score"),
        ("predict", "predict"),
    )

    payload = ransac_metadata_estimator_payload(estimator, method_mapping)

    assert payload == {"estimator": estimator, "method_mapping": method_mapping}
    assert payload["estimator"] is estimator
    assert payload["method_mapping"] is method_mapping


def test_ransac_metadata_estimator_payload_accepts_single_add_mapping() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_metadata_routing_shell import ransac_metadata_estimator_payload

    estimator = object()
    method_mapping = {"caller": "fit", "callee": "score"}

    payload = ransac_metadata_estimator_payload(estimator, method_mapping)

    assert payload["estimator"] is estimator
    assert payload["method_mapping"] is method_mapping


def test_ransac_metadata_router_result_preserves_router_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_metadata_routing_shell import ransac_metadata_router_result

    router = object()

    assert ransac_metadata_router_result(router) is router


def test_ransac_metadata_routing_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac_metadata_routing_shell import (
        ransac_metadata_estimator_payload,
        ransac_metadata_method_mapping_add_kwargs,
        ransac_metadata_router_owner,
        ransac_metadata_router_result,
    )

    with pytest.raises(ViolationError):
        ransac_metadata_router_owner("")

    with pytest.raises(ViolationError):
        ransac_metadata_method_mapping_add_kwargs("predict", "score")

    with pytest.raises(ViolationError):
        ransac_metadata_estimator_payload(object(), (("fit", "split"),))

    with pytest.raises(ViolationError):
        ransac_metadata_router_result(None)
