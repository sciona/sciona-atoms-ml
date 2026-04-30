from __future__ import annotations

import json
from pathlib import Path


FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/multiclass/output_code_book")
REFERENCES_PATH = FAMILY_DIR / "references.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.multiclass.output_code_book.output_code_uniform_book",
    "sciona.atoms.ml.sklearn.multiclass.output_code_book.output_code_discrete_book",
}


def test_output_code_book_references_cover_expected_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    atom_keys = {key.split("@", 1)[0] for key in payload["atoms"].keys()}
    assert atom_keys == EXPECTED_ATOMS


def test_output_code_book_references_use_known_ref_ids() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom_entry in payload["atoms"].values():
        ref_ids = {reference["ref_id"] for reference in atom_entry["references"]}
        assert ref_ids == {"repo_sklearn"}


def test_output_code_book_reference_locations_point_to_atoms_module() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for fqdn in payload["atoms"].keys():
        assert "@sciona/atoms/ml/sklearn/multiclass/output_code_book/atoms.py:" in fqdn
