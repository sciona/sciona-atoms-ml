from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "multioutput" / "postfit_attributes" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.multioutput.postfit_attributes.multioutput_partial_fit_n_features_in_update_required",
    "sciona.atoms.ml.sklearn.multioutput.postfit_attributes.multioutput_partial_fit_feature_names_in_update_required",
    "sciona.atoms.ml.sklearn.multioutput.postfit_attributes.multioutput_fit_n_features_in",
    "sciona.atoms.ml.sklearn.multioutput.postfit_attributes.multioutput_fit_feature_names_in",
    "sciona.atoms.ml.sklearn.multioutput.postfit_attributes.multioutput_classifier_classes",
}


def test_multioutput_postfit_attributes_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_multioutput_postfit_attributes_ref_ids_exist_and_have_metadata() -> None:
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


def test_multioutput_postfit_attributes_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.multioutput.postfit_attributes.atoms")
    leaf_names = {
        "multioutput_partial_fit_n_features_in_update_required",
        "multioutput_partial_fit_feature_names_in_update_required",
        "multioutput_fit_n_features_in",
        "multioutput_fit_feature_names_in",
        "multioutput_classifier_classes",
    }
    assert len(leaf_names) == 5
