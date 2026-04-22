from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import VALID_ACCEPTABILITY_BANDS


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data/review_bundles/ml_sklearn_neural_network_rbm.review_bundle.json"
MANIFEST_PATH = ROOT / "data/audit_manifest.json"
EXPECTED_ATOM_NAMES = {
    "sciona.atoms.ml.sklearn.neural_network.bernoulli_rbm_fit",
    "sciona.atoms.ml.sklearn.neural_network.bernoulli_rbm_free_energy",
    "sciona.atoms.ml.sklearn.neural_network.bernoulli_rbm_gibbs",
    "sciona.atoms.ml.sklearn.neural_network.bernoulli_rbm_mean_hiddens",
    "sciona.atoms.ml.sklearn.neural_network.bernoulli_rbm_partial_fit",
    "sciona.atoms.ml.sklearn.neural_network.bernoulli_rbm_sample_hiddens",
    "sciona.atoms.ml.sklearn.neural_network.bernoulli_rbm_sample_visibles",
    "sciona.atoms.ml.sklearn.neural_network.bernoulli_rbm_score_samples",
}


def test_bernoulli_rbm_review_bundle_exists_and_parses() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    assert bundle["schema_version"] == "1.0"
    assert bundle["provider_repo"] == "sciona-atoms-ml"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_record_path"] == "data/review_bundles/ml_sklearn_neural_network_rbm.review_bundle.json"
    assert {row["atom_name"] for row in bundle["rows"]} == EXPECTED_ATOM_NAMES


def test_bernoulli_rbm_review_bundle_rows_are_publishable() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in bundle["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
        assert row["trust_readiness"] in {"catalog_ready", "reviewed_with_limits"}
        assert row["overall_verdict"] == "pass"
        assert row["structural_status"] == "pass"
        assert row["semantic_status"] == "pass"
        assert row["runtime_status"] == "pass_with_limits"
        assert row["acceptability_band"] in VALID_ACCEPTABILITY_BANDS
        assert isinstance(row["risk_score"], int)
        assert isinstance(row["acceptability_score"], int)
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert not row["trust_blockers"]
        for source_path in row["source_paths"]:
            assert (ROOT / source_path).exists()


def test_manifest_contains_bernoulli_rbm_atoms_after_merge() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = {
        atom["atom_name"]: atom
        for atom in manifest.get("atoms", [])
        if atom.get("atom_name") in EXPECTED_ATOM_NAMES
    }
    assert set(entries) == EXPECTED_ATOM_NAMES
    for entry in entries.values():
        assert entry.get("review_status") in {"approved", "reviewed", "reviewed_pending"}
        assert entry.get("references_status") == "pass"
        assert entry.get("argument_details")
        assert entry.get("return_annotation")
