from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "ml_sklearn_decomposition_dictionary_learning_minibatch_partial_fit_shell.review_bundle.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "dictionary_learning_minibatch_partial_fit_shell" / "cdg.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_first_call",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_reset_required",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_initial_n_steps",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_initial_inner_stats",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_initial_dictionary",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_existing_dictionary",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_components",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_updated_n_steps",
}


def test_bundle_exists_and_has_dictionary_learning_minibatch_partial_fit_shell_atoms() -> None:
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
    family_node = cdg["nodes"][0]
    assert family_node["node_id"] == "sklearn_decomposition_dictionary_learning_minibatch_partial_fit_shell_root"
    assert bundle["family_batch"] == "ml_sklearn_decomposition_dictionary_learning_minibatch_partial_fit_shell"
    assert bundle["review_record_path"] == "data/review_bundles/ml_sklearn_decomposition_dictionary_learning_minibatch_partial_fit_shell.review_bundle.json"
