from __future__ import annotations

import json
from pathlib import Path


FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/decomposition/lda_fit_bookkeeping")
REFERENCES_PATH = FAMILY_DIR / "references.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_fit_use_online_batches",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_fit_evaluate_iteration_due",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_fit_converged",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_batch_bounds",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_partial_fit_first_call",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_partial_fit_require_matching_feature_count",
    "sciona.atoms.ml.sklearn.decomposition.lda_fit_bookkeeping.lda_check_nonnegative_dtype_names",
}


def test_lda_fit_bookkeeping_references_cover_expected_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    atom_keys = {
        key.split("@", 1)[0]
        for key in payload["atoms"].keys()
    }
    assert atom_keys == EXPECTED_ATOMS


def test_lda_fit_bookkeeping_references_use_known_ref_ids() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom_entry in payload["atoms"].values():
        ref_ids = {reference["ref_id"] for reference in atom_entry["references"]}
        assert "repo_sklearn" in ref_ids
        assert "pedregosa2011sklearn" in ref_ids


def test_lda_fit_bookkeeping_reference_locations_point_to_atoms_module() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for fqdn in payload["atoms"].keys():
        assert "@sciona/atoms/ml/sklearn/decomposition/lda_fit_bookkeeping/atoms.py:" in fqdn
