from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "inspection" / "partial_dependence_input_bookkeeping" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping.partial_dependence_feature_name_index",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping.partial_dependence_nonnegative_int_features",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping.partial_dependence_default_categorical_mask",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping.partial_dependence_boolean_categorical_mask",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping.partial_dependence_index_categorical_mask",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping.partial_dependence_name_categorical_mask",
}


def test_partial_dependence_input_bookkeeping_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_partial_dependence_input_bookkeeping_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]
    for atom_entry in payload["atoms"].values():
        refs = atom_entry["references"]
        assert refs
        for ref in refs:
            ref_id = ref["ref_id"]
            assert ref_id in registry
            assert ref["match_metadata"]["confidence"] == "high"


def test_partial_dependence_input_bookkeeping_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.inspection.partial_dependence_input_bookkeeping.atoms")
    observed = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert observed == {
        "partial_dependence_feature_name_index",
        "partial_dependence_nonnegative_int_features",
        "partial_dependence_default_categorical_mask",
        "partial_dependence_boolean_categorical_mask",
        "partial_dependence_index_categorical_mask",
        "partial_dependence_name_categorical_mask",
    }
