from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "multioutput" / "chain_fit_cv_updates" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_cv_updates.chain_fit_cv_update_required",
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_cv_updates.chain_fit_feature_column_index",
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_cv_updates.chain_fit_dense_cv_feature_update",
    "sciona.atoms.ml.sklearn.multioutput.chain_fit_cv_updates.chain_fit_sparse_cv_feature_update",
}


def test_chain_fit_cv_updates_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_chain_fit_cv_updates_ref_ids_exist_and_have_metadata() -> None:
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


def test_chain_fit_cv_updates_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.multioutput.chain_fit_cv_updates.atoms")
    leaf_names = {
        "chain_fit_cv_update_required",
        "chain_fit_feature_column_index",
        "chain_fit_dense_cv_feature_update",
        "chain_fit_sparse_cv_feature_update",
    }
    assert len(leaf_names) == 4
