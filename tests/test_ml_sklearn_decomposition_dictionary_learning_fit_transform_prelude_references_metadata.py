from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "dictionary_learning_fit_transform_prelude" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_fit_transform_prelude.dictionary_learning_fit_transform_require_positive_coding_supported",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_fit_transform_prelude.dictionary_learning_fit_transform_method",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_fit_transform_prelude.dictionary_learning_fit_transform_validated_data",
    "sciona.atoms.ml.sklearn.decomposition.dictionary_learning_fit_transform_prelude.dictionary_learning_fit_transform_n_components",
}


def test_dictionary_learning_fit_transform_prelude_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    fqdn_prefixes = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert fqdn_prefixes == EXPECTED_FQDNS


def test_dictionary_learning_fit_transform_prelude_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for ref_id, meta in payload["atoms"].items():
        assert ref_id.split("@", 1)[0] in EXPECTED_FQDNS
        refs = meta["references"]
        assert refs
        assert refs[0]["ref_id"] == "repo_sklearn"


def test_dictionary_learning_fit_transform_prelude_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.decomposition.dictionary_learning_fit_transform_prelude.atoms")
    package = import_module("sciona.atoms.ml.sklearn.decomposition.dictionary_learning_fit_transform_prelude")
    for leaf_name in {name.rsplit(".", 1)[-1] for name in EXPECTED_FQDNS}:
        assert hasattr(package, leaf_name)
