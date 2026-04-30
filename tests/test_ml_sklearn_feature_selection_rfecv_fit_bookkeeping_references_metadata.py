from __future__ import annotations

import json
from pathlib import Path


REFERENCES_PATH = Path("src/sciona/atoms/ml/sklearn/feature_selection/rfecv_fit_bookkeeping/references.json")

EXPECTED = {
    "sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping.rfecv_warn_min_features_too_large",
    "sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping.rfecv_resolved_min_features_to_select",
    "sciona.atoms.ml.sklearn.feature_selection.rfecv_fit_bookkeeping.rfecv_default_scoring_name",
}


def test_rfecv_fit_bookkeeping_references_cover_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    found = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert found == EXPECTED


def test_rfecv_fit_bookkeeping_references_use_known_sources() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom in payload["atoms"].values():
        ref_ids = {reference["ref_id"] for reference in atom["references"]}
        assert "repo_sklearn" in ref_ids
        assert "pedregosa2011sklearn" in ref_ids
