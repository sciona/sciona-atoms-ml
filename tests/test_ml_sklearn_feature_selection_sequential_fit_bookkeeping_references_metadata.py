from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "feature_selection" / "sequential_fit_bookkeeping" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping.sequential_resolve_n_features_to_select",
    "sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping.sequential_direction_tol_valid",
    "sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping.sequential_auto_select_enabled",
    "sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping.sequential_iteration_count",
    "sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping.sequential_tolerance_break",
    "sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping.sequential_finalize_support",
}


def test_sequential_fit_bookkeeping_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_sequential_fit_bookkeeping_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]
    for atom_entry in payload["atoms"].values():
        refs = atom_entry["references"]
        assert refs
        for ref in refs:
            ref_id = ref["ref_id"]
            assert ref_id in registry
            assert ref["match_metadata"]["confidence"] == "high"


def test_sequential_fit_bookkeeping_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping.atoms")
    observed = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert observed == {
        "sequential_resolve_n_features_to_select",
        "sequential_direction_tol_valid",
        "sequential_auto_select_enabled",
        "sequential_iteration_count",
        "sequential_tolerance_break",
        "sequential_finalize_support",
    }
