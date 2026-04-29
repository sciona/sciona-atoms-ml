from __future__ import annotations

import json
from pathlib import Path


BUNDLE_PATH = Path("data/review_bundles/ml_sklearn_decomposition_lda_fit_bookkeeping.review_bundle.json")

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_fit_use_online_batches",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_fit_evaluate_iteration_due",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_fit_converged",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_batch_bounds",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_partial_fit_first_call",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_partial_fit_require_matching_feature_count",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_check_nonnegative_dtype_names",
}


def test_lda_fit_bookkeeping_review_bundle_rows_cover_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    row_atoms = {row["atom_key"] for row in payload["rows"]}
    assert row_atoms == EXPECTED_ATOMS


def test_lda_fit_bookkeeping_review_bundle_source_paths_are_family_local() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    for row in payload["rows"]:
        for source_path in row["source_paths"]:
            assert source_path.startswith("src/sciona/atoms/ml/sklearn/decomposition/lda_fit_bookkeeping/") or source_path == "tests/test_ml_sklearn_decomposition_lda_fit_bookkeeping_behavior.py"


def test_lda_fit_bookkeeping_review_bundle_is_reviewed() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert payload["review_status"] == "reviewed"
    assert payload["review_semantic_verdict"] == "pass_with_limits"
