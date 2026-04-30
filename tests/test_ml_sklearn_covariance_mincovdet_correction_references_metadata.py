from __future__ import annotations

import json
from pathlib import Path


REFERENCES_PATH = Path("src/sciona/atoms/ml/sklearn/covariance/mincovdet_correction/references.json")

EXPECTED = {
    "sciona.atoms.ml.sklearn.covariance.mincovdet_correction.mincovdet_correct_covariance_guard",
    "sciona.atoms.ml.sklearn.covariance.mincovdet_correction.mincovdet_empirical_correction_factor",
    "sciona.atoms.ml.sklearn.covariance.mincovdet_correction.mincovdet_corrected_covariance",
    "sciona.atoms.ml.sklearn.covariance.mincovdet_correction.mincovdet_corrected_distances",
}


def test_mincovdet_correction_references_cover_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    found = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert found == EXPECTED


def test_mincovdet_correction_references_use_known_sources() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for key, atom in payload["atoms"].items():
        ref_ids = {reference["ref_id"] for reference in atom["references"]}
        assert "repo_sklearn" in ref_ids
        assert "pedregosa2011sklearn" in ref_ids
        if key.split("@", 1)[0].endswith("mincovdet_empirical_correction_factor"):
            assert "rousseeuw1999fastmcd" in ref_ids
