from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "multioutput" / "chain_fit_step_data" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_step_data.chain_fit_step_feature_limit",
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_step_data.chain_fit_target_column",
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_step_data.chain_fit_dense_step_features",
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_step_data.chain_fit_sparse_step_features",
}


def test_chain_fit_step_data_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_chain_fit_step_data_ref_ids_exist_and_have_metadata() -> None:
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


def test_chain_fit_step_data_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.multioutput.chain_fit_step_data.atoms")
    leaf_names = {
        "chain_fit_step_feature_limit",
        "chain_fit_target_column",
        "chain_fit_dense_step_features",
        "chain_fit_sparse_step_features",
    }
    assert len(leaf_names) == 4
