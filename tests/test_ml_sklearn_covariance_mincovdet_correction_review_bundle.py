from __future__ import annotations

import json
from pathlib import Path


BUNDLE_PATH = Path("data/review_bundles/ml_sklearn_covariance_mincovdet_correction.review_bundle.json")

EXPECTED = {
    "sciona.atoms.ml.sklearn.covariance.mincovdet_correction.mincovdet_correct_covariance_guard",
    "sciona.atoms.ml.sklearn.covariance.mincovdet_correction.mincovdet_empirical_correction_factor",
    "sciona.atoms.ml.sklearn.covariance.mincovdet_correction.mincovdet_corrected_covariance",
    "sciona.atoms.ml.sklearn.covariance.mincovdet_correction.mincovdet_corrected_distances",
}


def test_mincovdet_correction_review_bundle_rows_cover_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert {row["atom_key"] for row in payload["rows"]} == EXPECTED


def test_mincovdet_correction_review_bundle_reviewed() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert payload["review_status"] == "reviewed"
    assert payload["review_semantic_verdict"] == "pass_with_limits"
