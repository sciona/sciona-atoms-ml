from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "multiclass" / "output_code_fit_bookkeeping" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.multiclass.output_code_fit_bookkeeping.output_code_fit_require_nonempty_classes",
    "sciona.atoms.ml.sklearn.multiclass.output_code_fit_bookkeeping.output_code_fit_estimator_count",
    "sciona.atoms.ml.sklearn.multiclass.output_code_fit_bookkeeping.output_code_fit_n_features_in",
    "sciona.atoms.ml.sklearn.multiclass.output_code_fit_bookkeeping.output_code_fit_feature_names_in",
}


def test_output_code_fit_bookkeeping_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_output_code_fit_bookkeeping_ref_ids_exist_and_have_metadata() -> None:
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


def test_output_code_fit_bookkeeping_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.multiclass.output_code_fit_bookkeeping.atoms")
    leaf_names = {
        "output_code_fit_require_nonempty_classes",
        "output_code_fit_estimator_count",
        "output_code_fit_n_features_in",
        "output_code_fit_feature_names_in",
    }
    assert len(leaf_names) == 4
