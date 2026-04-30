from __future__ import annotations

import json
from pathlib import Path


BUNDLE_PATH = Path("data/review_bundles/ml_sklearn_decomposition_lda_em_updates.review_bundle.json")

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.decomposition.lda_em_updates.lda_e_step_document_topic_matrix",
    "sciona.atoms.ml.sklearn.decomposition.lda_em_updates.lda_e_step_sufficient_statistics",
    "sciona.atoms.ml.sklearn.decomposition.lda_em_updates.lda_online_update_weight",
    "sciona.atoms.ml.sklearn.decomposition.lda_em_updates.lda_online_document_ratio",
    "sciona.atoms.ml.sklearn.decomposition.lda_em_updates.lda_batch_components",
    "sciona.atoms.ml.sklearn.decomposition.lda_em_updates.lda_online_components",
}


def test_lda_em_updates_review_bundle_rows_cover_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    row_atoms = {row["atom_key"] for row in payload["rows"]}
    assert row_atoms == EXPECTED_ATOMS


def test_lda_em_updates_review_bundle_source_paths_are_family_local() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    for row in payload["rows"]:
        for source_path in row["source_paths"]:
            assert source_path.startswith("src/sciona/atoms/ml/sklearn/decomposition/lda_em_updates/") or source_path == "tests/test_ml_sklearn_decomposition_lda_em_updates_behavior.py"


def test_lda_em_updates_review_bundle_is_reviewed() -> None:
    payload = json.loads(BUNDLE_PATH.read_text())
    assert payload["review_status"] == "reviewed"
    assert payload["review_semantic_verdict"] == "pass_with_limits"
