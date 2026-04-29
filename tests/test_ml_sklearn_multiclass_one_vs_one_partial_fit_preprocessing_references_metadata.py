from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "multiclass" / "one_vs_one_partial_fit_preprocessing" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.multiclass.one_vs_one_partial_fit_preprocessing.one_vs_one_partial_fit_estimator_count",
    "sciona.atoms.ml.sklearn.multiclass.one_vs_one_partial_fit_preprocessing.one_vs_one_partial_fit_unknown_classes",
    "sciona.atoms.ml.sklearn.multiclass.one_vs_one_partial_fit_preprocessing.one_vs_one_partial_fit_pair_mask",
    "sciona.atoms.ml.sklearn.multiclass.one_vs_one_partial_fit_preprocessing.one_vs_one_partial_fit_subset_indices",
    "sciona.atoms.ml.sklearn.multiclass.one_vs_one_partial_fit_preprocessing.one_vs_one_partial_fit_binary_targets",
}


def test_one_vs_one_partial_fit_preprocessing_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_one_vs_one_partial_fit_preprocessing_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]
    for atom_entry in payload["atoms"].values():
        refs = atom_entry["references"]
        assert refs
        for ref in refs:
            ref_id = ref["ref_id"]
            assert ref_id in registry
            assert ref["match_metadata"]["confidence"] == "high"


def test_one_vs_one_partial_fit_preprocessing_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.multiclass.one_vs_one_partial_fit_preprocessing.atoms")
    observed = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert observed == {
        "one_vs_one_partial_fit_estimator_count",
        "one_vs_one_partial_fit_unknown_classes",
        "one_vs_one_partial_fit_pair_mask",
        "one_vs_one_partial_fit_subset_indices",
        "one_vs_one_partial_fit_binary_targets",
    }
