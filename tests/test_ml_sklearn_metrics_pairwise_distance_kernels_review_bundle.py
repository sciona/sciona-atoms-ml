from __future__ import annotations

import json
from pathlib import Path


BUNDLE_PATH = Path(
    "data/review_bundles/ml_sklearn_metrics_pairwise_distance_kernels.review_bundle.json"
)

EXPECTED = {
    "sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels.pairwise_rbf_kernel",
    "sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels.pairwise_additive_chi2_kernel",
    "sciona.atoms.ml.sklearn.metrics_pairwise_distance_kernels.pairwise_chi2_kernel",
}


def test_pairwise_distance_kernel_review_bundle_rows_cover_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert {row["atom_key"] for row in payload["rows"]} == EXPECTED


def test_pairwise_distance_kernel_review_bundle_reviewed() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert payload["review_status"] == "reviewed"
    assert payload["review_semantic_verdict"] == "pass_with_limits"
