from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "multioutput" / "response_method_bookkeeping" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.multioutput.response_method_bookkeeping.response_method_candidates",
    "sciona.atoms.ml.sklearn.multioutput.response_method_bookkeeping.response_method_name",
    "sciona.atoms.ml.sklearn.multioutput.response_method_bookkeeping.chain_fit_chain_method_name",
    "sciona.atoms.ml.sklearn.multioutput.response_method_bookkeeping.multioutput_predict_proba_available_from_estimators",
    "sciona.atoms.ml.sklearn.multioutput.response_method_bookkeeping.multioutput_predict_proba_available_from_base_estimator",
}


def test_multioutput_response_method_bookkeeping_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_multioutput_response_method_bookkeeping_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for entry in payload["atoms"].values():
        references = entry["references"]
        assert references
        for ref in references:
            assert ref["ref_id"]
            match_metadata = ref["match_metadata"]
            assert match_metadata["match_type"]
            assert match_metadata["confidence"]
            assert match_metadata["notes"]


def test_multioutput_response_method_bookkeeping_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.multioutput.response_method_bookkeeping.atoms")
    leaf_names = {
        "response_method_candidates",
        "response_method_name",
        "chain_fit_chain_method_name",
        "multioutput_predict_proba_available_from_estimators",
        "multioutput_predict_proba_available_from_base_estimator",
    }
    assert len(leaf_names) == 5
