from __future__ import annotations

import json
from pathlib import Path


FAMILY_DIR = Path("src/sciona/atoms/ml/sklearn/manifold/tsne_fit_preflight")
REFERENCES_PATH = FAMILY_DIR / "references.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight.tsne_fit_require_perplexity_below_sample_count",
    "sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight.tsne_fit_accept_sparse_formats",
    "sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight.tsne_fit_require_sparse_input_init_not_pca",
    "sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight.tsne_fit_require_precomputed_square_matrix",
    "sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight.tsne_fit_require_dense_exact_precomputed",
    "sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight.tsne_fit_require_barnes_hut_components",
}


def test_tsne_fit_preflight_references_cover_expected_atoms() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    atom_keys = {key.split("@", 1)[0] for key in payload["atoms"].keys()}
    assert atom_keys == EXPECTED_ATOMS


def test_tsne_fit_preflight_references_use_known_ref_ids() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom_key, atom_entry in payload["atoms"].items():
        ref_ids = {reference["ref_id"] for reference in atom_entry["references"]}
        assert "repo_sklearn" in ref_ids
        if atom_key.endswith("tsne_fit_require_perplexity_below_sample_count"):
            assert "vandermaaten2008tsne" in ref_ids


def test_tsne_fit_preflight_reference_locations_point_to_atoms_module() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for fqdn in payload["atoms"].keys():
        assert "@sciona/atoms/ml/sklearn/manifold/tsne_fit_preflight/atoms.py:" in fqdn
