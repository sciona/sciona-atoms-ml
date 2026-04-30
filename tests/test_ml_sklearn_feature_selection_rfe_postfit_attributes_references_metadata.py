from __future__ import annotations

import json
from pathlib import Path


FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/feature_selection/rfe_postfit_attributes")
REFERENCES_PATH = FAMILY_DIR / "references.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.feature_selection.rfe_postfit_attributes.rfe_estimator_type",
    "sciona.atoms.ml.sklearn.feature_selection.rfe_postfit_attributes.rfe_classes",
    "sciona.atoms.ml.sklearn.feature_selection.rfe_postfit_attributes.rfe_support_mask",
}


def test_rfe_postfit_attributes_references_cover_expected_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    atom_keys = {key.split("@", 1)[0] for key in payload["atoms"].keys()}
    assert atom_keys == EXPECTED_ATOMS


def test_rfe_postfit_attributes_references_use_known_ref_ids() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom_entry in payload["atoms"].values():
        ref_ids = {reference["ref_id"] for reference in atom_entry["references"]}
        assert "repo_sklearn" in ref_ids
        assert "pedregosa2011sklearn" in ref_ids


def test_rfe_postfit_attributes_reference_locations_point_to_atoms_module() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for fqdn in payload["atoms"].keys():
        assert "@sciona/atoms/ml/sklearn/feature_selection/rfe_postfit_attributes/atoms.py:" in fqdn
