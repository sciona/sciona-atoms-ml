from __future__ import annotations

import json
from pathlib import Path


BUNDLE_PATH = Path("data/review_bundles/ml_sklearn_multioutput_partial_fit_bookkeeping.review_bundle.json")

EXPECTED = {
    "sciona.atoms.ml.sklearn.multioutput.partial_fit_bookkeeping.multioutput_partial_fit_first_call",
    "sciona.atoms.ml.sklearn.multioutput.partial_fit_bookkeeping.multioutput_partial_fit_use_base_estimator",
    "sciona.atoms.ml.sklearn.multioutput.partial_fit_bookkeeping.multioutput_partial_fit_class_vector",
}


def test_multioutput_partial_fit_bookkeeping_review_bundle_rows_cover_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert {row["atom_key"] for row in payload["rows"]} == EXPECTED


def test_multioutput_partial_fit_bookkeeping_review_bundle_reviewed() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert payload["review_status"] == "reviewed"
    assert payload["review_semantic_verdict"] == "pass_with_limits"
