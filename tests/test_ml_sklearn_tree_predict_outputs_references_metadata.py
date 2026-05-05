from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "tree" / "predict_outputs" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.tree.predict_outputs.tree_classifier_single_output_labels",
    "sciona.atoms.ml.sklearn.tree.predict_outputs.tree_classifier_multioutput_labels",
    "sciona.atoms.ml.sklearn.tree.predict_outputs.tree_regressor_single_output_values",
    "sciona.atoms.ml.sklearn.tree.predict_outputs.tree_regressor_multioutput_values",
}


def test_tree_predict_outputs_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    actual = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert actual == EXPECTED_FQDNS


def test_tree_predict_outputs_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    for atom_id, atom_payload in payload["atoms"].items():
        assert atom_id.split("@", 1)[0] in EXPECTED_FQDNS
        references = atom_payload["references"]
        assert references
        for reference in references:
            assert reference["ref_id"]
            metadata = reference["match_metadata"]
            assert metadata["match_type"].startswith("manual")
            assert metadata["confidence"] in {"high", "medium"}
            assert metadata["notes"]


def test_tree_predict_outputs_atom_leaf_names_are_registered() -> None:
    module = import_module("sciona.atoms.ml.sklearn.tree.predict_outputs.atoms")
    for fqdn in EXPECTED_FQDNS:
        leaf_name = fqdn.rsplit(".", 1)[-1]
        assert hasattr(module, leaf_name)

