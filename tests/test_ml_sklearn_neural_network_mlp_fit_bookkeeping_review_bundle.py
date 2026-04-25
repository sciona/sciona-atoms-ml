from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import (
    VALID_ACCEPTABILITY_BANDS,
    VALID_PARITY_COVERAGE_LEVELS,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "ml_sklearn_neural_network_mlp_fit_bookkeeping.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "neural_network" / "mlp_fit_bookkeeping" / "cdg.json"
MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"

EXPECTED_ATOM_NAMES = {
    "sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping.mlp_batch_size",
    "sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping.mlp_batch_size_warning_required",
    "sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping.mlp_first_pass_required",
    "sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping.mlp_hidden_layer_sizes",
    "sciona.atoms.ml.sklearn.neural_network.mlp_fit_bookkeeping.mlp_partial_fit_require_no_early_stopping",
}


def _bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def test_bundle_exists_and_has_mlp_fit_bookkeeping_atoms() -> None:
    assert BUNDLE_PATH.exists()
    bundle = _bundle()
    assert len(bundle["rows"]) == 5
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_ATOM_NAMES


def test_bundle_level_fields() -> None:
    bundle = _bundle()
    assert bundle["provider_repo"] == "sciona-atoms-ml"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass_with_limits"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "reviewed_with_limits"
    assert bundle["review_record_path"] == "data/review_bundles/ml_sklearn_neural_network_mlp_fit_bookkeeping.review_bundle.json"


def test_each_row_has_review_metadata_and_existing_sources() -> None:
    bundle = _bundle()
    for row in bundle["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] == "pass_with_limits"
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["review_record_path"] == bundle["review_record_path"]
        assert row["source_paths"]
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists(), f"missing: {rel}"


def test_scores_and_enums_are_db_compatible() -> None:
    bundle = _bundle()
    for row in bundle["rows"]:
        assert isinstance(row["risk_score"], int)
        assert 0 <= row["risk_score"] <= 100
        assert isinstance(row["acceptability_score"], int)
        assert 0 <= row["acceptability_score"] <= 100
        assert row["acceptability_band"] in VALID_ACCEPTABILITY_BANDS
        assert row["parity_coverage_level"] in VALID_PARITY_COVERAGE_LEVELS


def test_cdg_atomic_nodes_have_publishable_io_specs() -> None:
    cdg = json.loads(CDG_PATH.read_text(encoding="utf-8"))
    atomic = [node for node in cdg["nodes"] if node.get("status") == "atomic"]
    expected_leaf_names = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_ATOM_NAMES}
    assert {node["name"] for node in atomic} == expected_leaf_names
    for node in atomic:
        assert node["node_id"] == node["name"]
        assert " " not in node["name"]
        assert node["inputs"]
        assert len(node["outputs"]) == 1
        assert node["outputs"][0]["name"] == "result"
        assert node["outputs"][0]["type_desc"]
        for item in node["inputs"]:
            assert item["name"]
            assert item["type_desc"]
            assert isinstance(item["required"], bool)


def test_manifest_contains_mlp_fit_bookkeeping_atoms_after_merge() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = {
        atom["atom_name"]: atom
        for atom in manifest.get("atoms", [])
        if atom.get("atom_name") in EXPECTED_ATOM_NAMES
    }
    assert set(entries) == EXPECTED_ATOM_NAMES
    for entry in entries.values():
        assert entry["review_status"] == "approved"
        assert entry["has_references"] is True
        assert entry["references_status"] == "pass"
        assert isinstance(entry["risk_score"], int)
        assert isinstance(entry["acceptability_score"], int)
        assert entry["acceptability_band"] in VALID_ACCEPTABILITY_BANDS
