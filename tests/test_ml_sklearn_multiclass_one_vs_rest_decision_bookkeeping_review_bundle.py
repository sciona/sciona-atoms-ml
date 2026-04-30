from __future__ import annotations

import json
from pathlib import Path


BUNDLE_PATH = Path("data/review_bundles/ml_sklearn_multiclass_one_vs_rest_decision_bookkeeping.review_bundle.json")

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.multiclass.one_vs_rest_decision_bookkeeping.one_vs_rest_decision_stack",
}


def test_one_vs_rest_decision_bookkeeping_review_bundle_rows_cover_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    row_atoms = {row["atom_key"] for row in payload["rows"]}
    assert row_atoms == EXPECTED_ATOMS


def test_one_vs_rest_decision_bookkeeping_review_bundle_source_paths_are_family_local() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    for row in payload["rows"]:
        for source_path in row["source_paths"]:
            assert source_path.startswith("src/sciona/atoms/ml/sklearn/multiclass/one_vs_rest_decision_bookkeeping/") or source_path == "tests/test_ml_sklearn_multiclass_one_vs_rest_decision_bookkeeping_behavior.py"


def test_one_vs_rest_decision_bookkeeping_review_bundle_is_reviewed() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert payload["review_status"] == "reviewed"
    assert payload["review_semantic_verdict"] == "pass_with_limits"
