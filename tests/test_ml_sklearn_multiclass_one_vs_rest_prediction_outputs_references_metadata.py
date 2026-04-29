from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "multiclass" / "one_vs_rest_prediction_outputs" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.multiclass.one_vs_rest_prediction_outputs.one_vs_rest_predict_maxima_init",
    "sciona.atoms.ml.sklearn.multiclass.one_vs_rest_prediction_outputs.one_vs_rest_predict_argmaxima_init",
    "sciona.atoms.ml.sklearn.multiclass.one_vs_rest_prediction_outputs.one_vs_rest_predict_multiclass_update",
    "sciona.atoms.ml.sklearn.multiclass.one_vs_rest_prediction_outputs.one_vs_rest_predict_labels_from_argmaxima",
}


def test_one_vs_rest_prediction_outputs_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_one_vs_rest_prediction_outputs_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]
    for atom_entry in payload["atoms"].values():
        refs = atom_entry["references"]
        assert refs
        for ref in refs:
            ref_id = ref["ref_id"]
            assert ref_id in registry
            assert ref["match_metadata"]["confidence"] == "high"


def test_one_vs_rest_prediction_outputs_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.multiclass.one_vs_rest_prediction_outputs.atoms")
    observed = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert observed == {
        "one_vs_rest_predict_maxima_init",
        "one_vs_rest_predict_argmaxima_init",
        "one_vs_rest_predict_multiclass_update",
        "one_vs_rest_predict_labels_from_argmaxima",
    }
