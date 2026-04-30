from __future__ import annotations

import json
from pathlib import Path


REFERENCES_PATH = Path(
    "src/sciona/atoms/ml/sklearn/multiclass/one_vs_rest_partial_fit_bookkeeping/references.json"
)

EXPECTED = {
    "sciona.atoms.ml.sklearn.multiclass.one_vs_rest_partial_fit_bookkeeping.one_vs_rest_partial_fit_estimator_count",
    "sciona.atoms.ml.sklearn.multiclass.one_vs_rest_partial_fit_bookkeeping.one_vs_rest_partial_fit_label_binarizer_classes",
    "sciona.atoms.ml.sklearn.multiclass.one_vs_rest_partial_fit_bookkeeping.one_vs_rest_partial_fit_n_features_in",
}


def test_one_vs_rest_partial_fit_bookkeeping_references_cover_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    found = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert found == EXPECTED


def test_one_vs_rest_partial_fit_bookkeeping_references_use_known_sources() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom in payload["atoms"].values():
        ref_ids = {reference["ref_id"] for reference in atom["references"]}
        assert "repo_sklearn" in ref_ids
        assert "pedregosa2011sklearn" in ref_ids
