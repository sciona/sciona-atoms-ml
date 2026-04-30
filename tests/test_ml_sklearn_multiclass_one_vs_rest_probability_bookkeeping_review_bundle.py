from __future__ import annotations

import json
from pathlib import Path


BUNDLE_PATH = Path("data/review_bundles/ml_sklearn_multiclass_one_vs_rest_probability_bookkeeping.review_bundle.json")

EXPECTED = {
    "sciona.atoms.ml.sklearn.multiclass.one_vs_rest_probability_bookkeeping.one_vs_rest_positive_probability_stack",
}


def test_one_vs_rest_probability_bookkeeping_review_bundle_rows_cover_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert {row["atom_key"] for row in payload["rows"]} == EXPECTED


def test_one_vs_rest_probability_bookkeeping_review_bundle_reviewed() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert payload["review_status"] == "reviewed"
    assert payload["review_semantic_verdict"] == "pass_with_limits"
