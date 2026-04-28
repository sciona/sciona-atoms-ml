from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "multioutput" / "chain_prediction_bookkeeping" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.multioutput.chain_prediction_bookkeeping.chain_prediction_method_name",
    "sciona.atoms.ml.sklearn.multioutput.chain_prediction_bookkeeping.chain_prediction_output_buffer",
    "sciona.atoms.ml.sklearn.multioutput.chain_prediction_bookkeeping.chain_prediction_feature_buffer",
    "sciona.atoms.ml.sklearn.multioutput.chain_prediction_bookkeeping.chain_prediction_previous_predictions",
    "sciona.atoms.ml.sklearn.multioutput.chain_prediction_bookkeeping.chain_sparse_hstack_base",
}


def test_chain_prediction_bookkeeping_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_chain_prediction_bookkeeping_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]
    for atom_entry in payload["atoms"].values():
        refs = atom_entry["references"]
        assert refs
        for ref in refs:
            ref_id = ref["ref_id"]
            assert ref_id in registry
            assert ref["match_metadata"]["confidence"] == "high"


def test_chain_prediction_bookkeeping_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.multioutput.chain_prediction_bookkeeping.atoms")
    observed = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert observed == {
        "chain_prediction_method_name",
        "chain_prediction_output_buffer",
        "chain_prediction_feature_buffer",
        "chain_prediction_previous_predictions",
        "chain_sparse_hstack_base",
    }
