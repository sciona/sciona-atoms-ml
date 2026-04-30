from __future__ import annotations

import json
from pathlib import Path


FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/manifold/tsne_nn_probabilities")
REFERENCES_PATH = FAMILY_DIR / "references.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.manifold.tsne_nn_probabilities.tsne_nn_distance_blocks",
    "sciona.atoms.ml.sklearn.manifold.tsne_nn_probabilities.tsne_nn_conditional_probability_matrix",
    "sciona.atoms.ml.sklearn.manifold.tsne_nn_probabilities.tsne_nn_joint_probabilities",
}


def test_tsne_nn_probabilities_references_cover_expected_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    atom_keys = {key.split("@", 1)[0] for key in payload["atoms"].keys()}
    assert atom_keys == EXPECTED_ATOMS


def test_tsne_nn_probabilities_references_use_known_ref_ids() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom_entry in payload["atoms"].values():
        ref_ids = {reference["ref_id"] for reference in atom_entry["references"]}
        assert "repo_sklearn" in ref_ids
        assert "vandermaaten2008tsne" in ref_ids


def test_tsne_nn_probabilities_reference_locations_point_to_atoms_module() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for fqdn in payload["atoms"].keys():
        assert "@sciona/atoms/ml/sklearn/manifold/tsne_nn_probabilities/atoms.py:" in fqdn
