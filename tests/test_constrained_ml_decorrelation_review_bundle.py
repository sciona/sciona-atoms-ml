from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import (
    VALID_ACCEPTABILITY_BANDS,
    VALID_PARITY_COVERAGE_LEVELS,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "ml_constrained_ml_decorrelation.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "constrained_ml" / "decorrelation" / "cdg.json"

EXPECTED_ATOM_NAMES = {
    "sciona.atoms.ml.constrained_ml.decorrelation.compute_cvm_mass_decorrelation",
    "sciona.atoms.ml.constrained_ml.decorrelation.compute_ks_agreement",
    "sciona.atoms.ml.constrained_ml.decorrelation.roc_auc_truncated_weighted",
    "sciona.atoms.ml.constrained_ml.decorrelation.noise_injection_decorrelation",
}


def _bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def test_bundle_exists_and_has_decorrelation_atoms() -> None:
    assert BUNDLE_PATH.exists()
    bundle = _bundle()
    assert len(bundle["rows"]) == 4
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_ATOM_NAMES


def test_bundle_level_fields() -> None:
    bundle = _bundle()
    assert bundle["provider_repo"] == "sciona-atoms-ml"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] in {"pass", "pass_with_limits"}
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "reviewed_with_limits"
    assert bundle["review_record_path"] == "data/review_bundles/ml_constrained_ml_decorrelation.review_bundle.json"


def test_each_row_has_review_metadata_and_existing_sources() -> None:
    bundle = _bundle()
    for row in bundle["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["overall_verdict"] in {"pass", "pass_with_limits"}
        assert row["has_references"] is True
        assert row["acceptability_band"] in VALID_ACCEPTABILITY_BANDS
        assert row["parity_coverage_level"] in VALID_PARITY_COVERAGE_LEVELS
        for path in row["source_paths"]:
            full = ROOT / path
            assert full.exists(), f"missing source path: {path}"


def test_cdg_nodes_cover_all_atoms() -> None:
    cdg = json.loads(CDG_PATH.read_text(encoding="utf-8"))
    atomic_names = {
        node["name"]
        for node in cdg["nodes"]
        if node.get("status") == "atomic"
    }
    expected_fn_names = {
        "compute_cvm_mass_decorrelation",
        "compute_ks_agreement",
        "roc_auc_truncated_weighted",
        "noise_injection_decorrelation",
    }
    assert expected_fn_names == atomic_names
