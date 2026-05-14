from __future__ import annotations

import json
from pathlib import Path


FAMILY = "coordinate_descent_elastic_net_tags_super_callback_shell"
MODULE = f"sciona.atoms.ml.sklearn.linear_model.{FAMILY}"
REVIEW_BUNDLE = Path(
    "data/review_bundles/"
    "ml_sklearn_linear_model_coordinate_descent_elastic_net_tags_super_callback_shell.review_bundle.json"
)

EXPECTED_ATOMS = {
    f"{MODULE}.cd_elastic_net_tags_super_result",
    f"{MODULE}.cd_elastic_net_tags_return",
}


def test_coordinate_descent_elastic_net_tags_super_callback_shell_review_bundle_shape() -> None:
    bundle = json.loads(REVIEW_BUNDLE.read_text())

    assert bundle["schema_version"] == "1.0"
    assert (
        bundle["bundle_id"]
        == "sciona.atoms.review_bundle.ml.sklearn_linear_model_coordinate_descent_elastic_net_tags_super_callback_shell.v1"
    )
    assert bundle["provider_repo"] == "sciona-atoms-ml"
    assert bundle["family_batch"] == "ml_sklearn_linear_model_coordinate_descent_elastic_net_tags_super_callback_shell"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass_with_limits"
    assert bundle["blocking_findings"] == []
    assert bundle["required_actions"] == []

    atom_keys = {row["atom_key"] for row in bundle["rows"]}
    assert atom_keys == EXPECTED_ATOMS


def test_coordinate_descent_elastic_net_tags_super_callback_shell_review_bundle_rows_are_catalog_ready() -> None:
    bundle = json.loads(REVIEW_BUNDLE.read_text())

    for row in bundle["rows"]:
        assert row["atom_name"] == row["atom_key"]
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass_with_limits"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["overall_verdict"] == "pass"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["parity_test_status"] == "pass"
        assert str(REVIEW_BUNDLE) == row["review_record_path"]
