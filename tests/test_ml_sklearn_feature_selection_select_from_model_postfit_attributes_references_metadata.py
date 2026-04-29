from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "feature_selection" / "select_from_model_postfit_attributes" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.feature_selection.select_from_model_postfit_attributes.select_from_model_partial_fit_first_call",
    "sciona.atoms.ml.sklearn.feature_selection.select_from_model_postfit_attributes.select_from_model_postfit_n_features_in",
    "sciona.atoms.ml.sklearn.feature_selection.select_from_model_postfit_attributes.select_from_model_postfit_feature_names_in",
}


def test_select_from_model_postfit_attributes_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_select_from_model_postfit_attributes_ref_ids_exist_and_have_metadata() -> None:
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


def test_select_from_model_postfit_attributes_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.feature_selection.select_from_model_postfit_attributes.atoms")
    leaf_names = {
        "select_from_model_partial_fit_first_call",
        "select_from_model_postfit_n_features_in",
        "select_from_model_postfit_feature_names_in",
    }
    assert len(leaf_names) == 3
