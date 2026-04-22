from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data/review_bundles/ml_sklearn_mixture_gaussian.review_bundle.json"
CDG_PATH = ROOT / "src/sciona/atoms/ml/sklearn/mixture/cdg.json"
MANIFEST_PATH = ROOT / "data/audit_manifest.json"
EXPECTED = {
    "sciona.atoms.ml.sklearn.mixture.gaussian_mixture_diag_fit",
    "sciona.atoms.ml.sklearn.mixture.gaussian_mixture_diag_score_samples",
    "sciona.atoms.ml.sklearn.mixture.gaussian_mixture_diag_score",
    "sciona.atoms.ml.sklearn.mixture.gaussian_mixture_diag_predict_proba",
    "sciona.atoms.ml.sklearn.mixture.gaussian_mixture_diag_predict",
    "sciona.atoms.ml.sklearn.mixture.gaussian_mixture_diag_bic",
    "sciona.atoms.ml.sklearn.mixture.gaussian_mixture_diag_aic",
    "sciona.atoms.ml.sklearn.mixture.bayesian_gaussian_mixture_diag_fit",
    "sciona.atoms.ml.sklearn.mixture.bayesian_gaussian_mixture_diag_score_samples",
    "sciona.atoms.ml.sklearn.mixture.bayesian_gaussian_mixture_diag_score",
    "sciona.atoms.ml.sklearn.mixture.bayesian_gaussian_mixture_diag_predict_proba",
    "sciona.atoms.ml.sklearn.mixture.bayesian_gaussian_mixture_diag_predict",
}


def test_gaussian_mixture_review_bundle_rows_are_publishable() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())
    assert bundle["provider_repo"] == "sciona-atoms-ml"
    assert bundle["review_status"] == "reviewed"
    assert bundle["trust_readiness"] in {"catalog_ready", "reviewed_with_limits"}
    rows = bundle["rows"]
    assert {row["atom_name"] for row in rows} == EXPECTED
    for row in rows:
        assert row["review_status"] == "reviewed"
        assert row["overall_verdict"] == "pass"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert isinstance(row["risk_score"], int)
        assert isinstance(row["acceptability_score"], int)
        for source in row["source_paths"]:
            assert (ROOT / source).exists()


def test_gaussian_mixture_cdg_nodes_have_io_specs() -> None:
    cdg = json.loads(CDG_PATH.read_text())
    atomic = [node for node in cdg["nodes"] if node.get("status") == "atomic"]
    assert {f"sciona.atoms.ml.sklearn.mixture.{node['name']}" for node in atomic} == EXPECTED
    for node in atomic:
        assert node["inputs"]
        assert node["outputs"] == [{"name": "result", "type_desc": node["outputs"][0]["type_desc"]}]


def test_manifest_contains_gaussian_mixture_atoms_after_merge() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    names = {atom["atom_name"] for atom in manifest["atoms"]}
    assert EXPECTED <= names
