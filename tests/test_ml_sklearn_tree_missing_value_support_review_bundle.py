from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "ml_sklearn_tree_missing_value_support.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "tree" / "missing_value_support" / "cdg.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.tree.missing_value_support.tree_missing_values_x_is_sparse",
    "sciona.atoms.ml.sklearn.tree.missing_value_support.tree_missing_values_allow_nan_enabled",
    "sciona.atoms.ml.sklearn.tree.missing_value_support.tree_missing_values_monotonic_constraints_absent",
    "sciona.atoms.ml.sklearn.tree.missing_value_support.tree_missing_values_supported",
}


def test_bundle_exists_and_has_tree_missing_value_support_atoms() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())
    assert bundle["bundle_id"] == "sciona.atoms.review_bundle.ml.sklearn_tree_missing_value_support.v1"
    assert bundle["family_batch"] == "ml_sklearn_tree_missing_value_support"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass_with_limits"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["review_record_path"] == "data/review_bundles/ml_sklearn_tree_missing_value_support.review_bundle.json"
    row_atoms = {row["atom_key"] for row in bundle["rows"]}
    assert row_atoms == EXPECTED_ATOMS


def test_bundle_rows_align_with_cdg_atoms() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())
    cdg = json.loads(CDG_PATH.read_text())
    atomic_names = {node["name"] for node in cdg["nodes"] if node["status"] == "atomic"}
    assert atomic_names == {atom.rsplit(".", 1)[-1] for atom in EXPECTED_ATOMS}
    for row in bundle["rows"]:
        assert row["overall_verdict"] == "pass"
        assert row["references_status"] == "pass"
        assert row["parity_test_status"] == "pass"

