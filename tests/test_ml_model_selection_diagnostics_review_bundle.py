from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import (
    VALID_ACCEPTABILITY_BANDS,
    VALID_PARITY_COVERAGE_LEVELS,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "ml_model_selection_diagnostics.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "model_selection" / "diagnostics" / "cdg.json"

EXPECTED_ATOM_NAMES = {
    "sciona.atoms.ml.model_selection.diagnostics.compute_condition_number",
    "sciona.atoms.ml.model_selection.diagnostics.compute_n_p_ratio",
    "sciona.atoms.ml.model_selection.diagnostics.compute_mutual_incoherence",
    "sciona.atoms.ml.model_selection.diagnostics.check_lasso_sample_complexity",
    "sciona.atoms.ml.model_selection.diagnostics.compute_excess_kurtosis",
    "sciona.atoms.ml.model_selection.diagnostics.compute_residual_kurtosis",
    "sciona.atoms.ml.model_selection.diagnostics.compute_dispersion_index",
    "sciona.atoms.ml.model_selection.diagnostics.estimate_tweedie_power",
    "sciona.atoms.ml.model_selection.diagnostics.estimate_noise_level",
    "sciona.atoms.ml.model_selection.diagnostics.count_categorical_features",
    "sciona.atoms.ml.model_selection.diagnostics.compute_skewness",
    "sciona.atoms.ml.model_selection.diagnostics.compute_vif",
    "sciona.atoms.ml.model_selection.diagnostics.test_normality",
    "sciona.atoms.ml.model_selection.diagnostics.is_sparse",
    "sciona.atoms.ml.model_selection.diagnostics.compute_explained_variance_ratio",
    "sciona.atoms.ml.model_selection.diagnostics.check_time_series_index",
}


def _bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def test_bundle_exists_and_has_diagnostic_atoms() -> None:
    assert BUNDLE_PATH.exists()
    bundle = _bundle()
    assert len(bundle["rows"]) == 16
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_ATOM_NAMES


def test_bundle_level_fields() -> None:
    bundle = _bundle()
    assert bundle["provider_repo"] == "sciona-atoms-ml"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] in {"pass", "pass_with_limits"}
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "reviewed_with_limits"
    assert bundle["review_record_path"] == "data/review_bundles/ml_model_selection_diagnostics.review_bundle.json"


def test_each_row_has_review_metadata_and_existing_sources() -> None:
    bundle = _bundle()
    for row in bundle["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        assert row["trust_readiness"] == "catalog_ready"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert row["review_record_path"] == bundle["review_record_path"]
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
    assert len(atomic) == 16
    for node in atomic:
        assert node["node_id"] == node["name"]
        assert node["inputs"]
        assert len(node["outputs"]) == 1
        assert node["outputs"][0]["name"] == "result"
        for item in node["inputs"]:
            assert item["name"]
            assert item["type_desc"]
            assert isinstance(item["required"], bool)
