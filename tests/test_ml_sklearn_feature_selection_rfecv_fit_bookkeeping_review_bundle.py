from __future__ import annotations

import json
from pathlib import Path


BUNDLE_PATH = Path("data/review_bundles/ml_sklearn_feature_selection_rfecv_fit_bookkeeping.review_bundle.json")

EXPECTED = {
    "sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping.rfecv_warn_min_features_too_large",
    "sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping.rfecv_resolved_min_features_to_select",
    "sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping.rfecv_default_scoring_name",
}


def test_rfecv_fit_bookkeeping_review_bundle_rows_cover_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert {row["atom_key"] for row in payload["rows"]} == EXPECTED


def test_rfecv_fit_bookkeeping_review_bundle_reviewed() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert payload["review_status"] == "reviewed"
    assert payload["review_semantic_verdict"] == "pass_with_limits"
