from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "ml_sklearn_naive_bayes_bernoulli.review_bundle.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.naive_bayes.bernoulli_nb_binarize",
    "sciona.atoms.ml.sklearn.naive_bayes.bernoulli_nb_count",
    "sciona.atoms.ml.sklearn.naive_bayes.bernoulli_nb_feature_log_prob",
    "sciona.atoms.ml.sklearn.naive_bayes.bernoulli_nb_fit",
    "sciona.atoms.ml.sklearn.naive_bayes.bernoulli_nb_joint_log_likelihood",
    "sciona.atoms.ml.sklearn.naive_bayes.bernoulli_nb_predict_log_proba",
    "sciona.atoms.ml.sklearn.naive_bayes.bernoulli_nb_predict_proba",
    "sciona.atoms.ml.sklearn.naive_bayes.bernoulli_nb_predict",
}


def test_bernoulli_nb_review_bundle_exists_and_parses() -> None:
    assert BUNDLE_PATH.exists()
    payload = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    assert payload["provider_repo"] == "sciona-atoms-ml"
    assert payload["review_status"] == "reviewed"
    assert payload["review_semantic_verdict"] in {"pass", "pass_with_limits"}
    assert payload["review_developer_semantic_verdict"] in {"pass", "pass_with_limits"}
    assert payload["trust_readiness"] in {"catalog_ready", "reviewed_with_limits"}


def test_bernoulli_nb_review_bundle_contains_expected_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    assert {row["atom_key"] for row in payload["rows"]} == EXPECTED_ATOMS


def test_bernoulli_nb_review_bundle_rows_are_publishable() -> None:
    payload = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        assert row["review_status"] == "reviewed"
        assert row["overall_verdict"] == "pass"
        assert row["structural_status"] == "pass"
        assert row["semantic_status"] == "pass"
        assert row["runtime_status"] in {"pass", "pass_with_limits"}
        assert row["developer_semantics_status"] in {"pass", "pass_with_limits"}
        assert row["trust_readiness"] in {"catalog_ready", "reviewed_with_limits"}
        assert row["has_references"] is True
        assert row["references_status"] == "pass"
        assert isinstance(row["risk_score"], int)
        assert isinstance(row["acceptability_score"], int)
        for source_path in row["source_paths"]:
            assert (ROOT / source_path).exists(), source_path
