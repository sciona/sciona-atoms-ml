from __future__ import annotations

import json
from pathlib import Path

from sciona.atoms.audit_review_bundles import (
    VALID_ACCEPTABILITY_BANDS,
    VALID_PARITY_COVERAGE_LEVELS,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "ml_sklearn_preprocessing_basic.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "preprocessing" / "cdg.json"
MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"

EXPECTED_ATOM_NAMES = {
    "sciona.atoms.ml.sklearn.preprocessing.add_dummy_feature",
    "sciona.atoms.ml.sklearn.preprocessing.binarize",
    "sciona.atoms.ml.sklearn.preprocessing.binarizer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.kbins_discretizer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.kbins_discretizer_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.kbins_discretizer_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.kbins_discretizer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.kernel_centerer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.kernel_centerer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarize",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarizer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarizer_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarizer_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_binarizer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_encoder_fit",
    "sciona.atoms.ml.sklearn.preprocessing.label_encoder_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_encoder_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.label_encoder_transform",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scale",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scaler_fit",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scaler_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scaler_partial_fit",
    "sciona.atoms.ml.sklearn.preprocessing.maxabs_scaler_transform",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scale",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scaler_fit",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scaler_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scaler_partial_fit",
    "sciona.atoms.ml.sklearn.preprocessing.minmax_scaler_transform",
    "sciona.atoms.ml.sklearn.preprocessing.multi_label_binarizer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.multi_label_binarizer_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.multi_label_binarizer_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.multi_label_binarizer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.normalize",
    "sciona.atoms.ml.sklearn.preprocessing.normalizer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.onehot_encoder_fit",
    "sciona.atoms.ml.sklearn.preprocessing.onehot_encoder_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.onehot_encoder_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.onehot_encoder_transform",
    "sciona.atoms.ml.sklearn.preprocessing.ordinal_encoder_fit",
    "sciona.atoms.ml.sklearn.preprocessing.ordinal_encoder_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.ordinal_encoder_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.ordinal_encoder_transform",
    "sciona.atoms.ml.sklearn.preprocessing.polynomial_features_fit",
    "sciona.atoms.ml.sklearn.preprocessing.polynomial_features_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.polynomial_features_transform",
    "sciona.atoms.ml.sklearn.preprocessing.power_transform",
    "sciona.atoms.ml.sklearn.preprocessing.power_transformer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.power_transformer_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.power_transformer_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.power_transformer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.quantile_transform",
    "sciona.atoms.ml.sklearn.preprocessing.quantile_transformer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.quantile_transformer_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.quantile_transformer_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.quantile_transformer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.robust_scale",
    "sciona.atoms.ml.sklearn.preprocessing.robust_scaler_fit",
    "sciona.atoms.ml.sklearn.preprocessing.robust_scaler_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.robust_scaler_transform",
    "sciona.atoms.ml.sklearn.preprocessing.scale",
    "sciona.atoms.ml.sklearn.preprocessing.spline_transformer_fit",
    "sciona.atoms.ml.sklearn.preprocessing.spline_transformer_fit_transform",
    "sciona.atoms.ml.sklearn.preprocessing.spline_transformer_transform",
    "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_fit",
    "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_inverse_transform",
    "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_partial_fit",
    "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_transform",
}


def _bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def test_bundle_exists_and_has_sixty_seven_atoms() -> None:
    assert BUNDLE_PATH.exists()
    bundle = _bundle()
    assert len(bundle["rows"]) == 67
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_ATOM_NAMES


def test_bundle_level_fields() -> None:
    bundle = _bundle()
    assert bundle["provider_repo"] == "sciona-atoms-ml"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] in {"pass", "pass_with_limits"}
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert bundle["trust_readiness"] == "reviewed_with_limits"
    assert bundle["review_record_path"] == "data/review_bundles/ml_sklearn_preprocessing_basic.review_bundle.json"


def test_each_row_has_review_metadata_and_existing_sources() -> None:
    bundle = _bundle()
    for row in bundle["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["review_semantic_verdict"] in {"pass", "pass_with_limits"}
        assert row["review_developer_semantic_verdict"] == "pass_with_limits"
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
    assert {node["name"] for node in atomic} == {
        "add_dummy_feature",
        "binarize",
        "binarizer_transform",
        "kbins_discretizer_fit",
        "kbins_discretizer_fit_transform",
        "kbins_discretizer_inverse_transform",
        "kbins_discretizer_transform",
        "kernel_centerer_fit",
        "kernel_centerer_transform",
        "label_binarize",
        "label_binarizer_fit",
        "label_binarizer_fit_transform",
        "label_binarizer_inverse_transform",
        "label_binarizer_transform",
        "label_encoder_fit",
        "label_encoder_fit_transform",
        "label_encoder_inverse_transform",
        "label_encoder_transform",
        "maxabs_scale",
        "maxabs_scaler_fit",
        "maxabs_scaler_inverse_transform",
        "maxabs_scaler_partial_fit",
        "maxabs_scaler_transform",
        "minmax_scale",
        "minmax_scaler_fit",
        "minmax_scaler_inverse_transform",
        "minmax_scaler_partial_fit",
        "minmax_scaler_transform",
        "multi_label_binarizer_fit",
        "multi_label_binarizer_fit_transform",
        "multi_label_binarizer_inverse_transform",
        "multi_label_binarizer_transform",
        "normalize",
        "normalizer_transform",
        "onehot_encoder_fit",
        "onehot_encoder_fit_transform",
        "onehot_encoder_inverse_transform",
        "onehot_encoder_transform",
        "ordinal_encoder_fit",
        "ordinal_encoder_fit_transform",
        "ordinal_encoder_inverse_transform",
        "ordinal_encoder_transform",
        "polynomial_features_fit",
        "polynomial_features_fit_transform",
        "polynomial_features_transform",
        "power_transform",
        "power_transformer_fit",
        "power_transformer_fit_transform",
        "power_transformer_inverse_transform",
        "power_transformer_transform",
        "quantile_transform",
        "quantile_transformer_fit",
        "quantile_transformer_fit_transform",
        "quantile_transformer_inverse_transform",
        "quantile_transformer_transform",
        "robust_scale",
        "robust_scaler_fit",
        "robust_scaler_inverse_transform",
        "robust_scaler_transform",
        "scale",
        "spline_transformer_fit",
        "spline_transformer_fit_transform",
        "spline_transformer_transform",
        "standard_scaler_fit",
        "standard_scaler_inverse_transform",
        "standard_scaler_partial_fit",
        "standard_scaler_transform",
    }
    for node in atomic:
        assert node["node_id"] == node["name"]
        assert node["inputs"]
        assert len(node["outputs"]) == 1
        assert node["outputs"][0]["name"] == "result"
        for item in node["inputs"]:
            assert item["name"]
            assert item["type_desc"]
            assert isinstance(item["required"], bool)


def test_manifest_contains_preprocessing_atoms_after_merge() -> None:
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
