from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "ml_sklearn_decomposition_sparse_coder_api_shell.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "sparse_coder_api_shell" / "cdg.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_fit_return_self",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_transform_dictionary",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_requires_fit_tag",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_preserves_dtype_tags",
    "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell.sparse_coder_n_features_out",
}


def test_bundle_exists_and_has_sparse_coder_api_shell_atoms() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())
    rows = bundle["rows"]
    atom_names = {row["atom_name"] for row in rows}
    assert atom_names == EXPECTED_ATOMS
    assert all(row["overall_verdict"] == "pass" for row in rows)
    assert all(row["structural_status"] == "pass" for row in rows)
    assert all(row["semantic_status"] == "pass" for row in rows)
    assert all(row["has_references"] for row in rows)


def test_bundle_matches_cdg_family() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())
    cdg = json.loads(CDG_PATH.read_text())
    assert cdg["module"] == "sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell"
    assert bundle["family_batch"] == "ml_sklearn_decomposition_sparse_coder_api_shell"
    assert bundle["review_record_path"] == "data/review_bundles/ml_sklearn_decomposition_sparse_coder_api_shell.review_bundle.json"
