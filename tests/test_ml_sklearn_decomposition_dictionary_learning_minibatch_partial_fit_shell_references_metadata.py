from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "dictionary_learning_minibatch_partial_fit_shell" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_first_call",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_reset_required",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_initial_n_steps",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_initial_inner_stats",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_initial_dictionary",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_existing_dictionary",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_components",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.dictionary_learning_minibatch_partial_fit_updated_n_steps",
}


def test_dictionary_learning_minibatch_partial_fit_shell_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    fqdn_prefixes = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert fqdn_prefixes == EXPECTED_FQDNS


def test_dictionary_learning_minibatch_partial_fit_shell_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for ref_id, meta in payload["atoms"].items():
        assert ref_id.split("@", 1)[0] in EXPECTED_FQDNS
        refs = meta["references"]
        assert refs
        assert refs[0]["ref_id"] == "repo_sklearn"


def test_dictionary_learning_minibatch_partial_fit_shell_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell.atoms")
    package = import_module("sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_partial_fit_shell")
    for leaf_name in {name.rsplit(".", 1)[-1] for name in EXPECTED_FQDNS}:
        assert hasattr(package, leaf_name)
