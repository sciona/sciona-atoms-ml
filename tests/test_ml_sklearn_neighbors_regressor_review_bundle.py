from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import VALID_ACCEPTABILITY_BANDS


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "ml_sklearn_neighbors_regressor.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "neighbors" / "cdg.json"
MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"

EXPECTED_ATOM_NAMES = {
    "sciona.atoms.ml.sklearn.neighbors.kneighbors_regressor_fit",
    "sciona.atoms.ml.sklearn.neighbors.kneighbors_regressor_predict",
    "sciona.atoms.ml.sklearn.neighbors.radius_neighbors_regressor_fit",
    "sciona.atoms.ml.sklearn.neighbors.radius_neighbors_regressor_predict",
}


def test_neighbors_regressor_review_bundle_exists_and_parses() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    assert bundle["provider_repo"] == "sciona-atoms-ml"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass_with_limits"
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_ATOM_NAMES


def test_neighbors_regressor_review_bundle_rows_are_publishable() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert isinstance(row["risk_score"], int)
        assert isinstance(row["acceptability_score"], int)
        assert row["acceptability_band"] in VALID_ACCEPTABILITY_BANDS
        for rel in row["source_paths"]:
            assert (ROOT / rel).exists()


def test_neighbors_regressor_cdg_nodes_have_io_specs() -> None:
    cdg = json.loads(CDG_PATH.read_text(encoding="utf-8"))
    nodes = {
        f"sciona.atoms.ml.sklearn.neighbors.{node['name']}": node
        for node in cdg["nodes"]
        if node.get("status") == "atomic"
    }
    assert EXPECTED_ATOM_NAMES.issubset(nodes)
    for fqdn in EXPECTED_ATOM_NAMES:
        node = nodes[fqdn]
        assert node["node_id"] == node["name"]
        assert node["inputs"]
        assert len(node["outputs"]) == 1
        assert node["outputs"][0]["name"] == "result"


def test_manifest_contains_neighbors_regressor_atoms_after_merge() -> None:
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
