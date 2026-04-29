from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "feature_selection" / "rfe_state_bookkeeping" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping.rfe_initial_support_mask",
    "sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping.rfe_initial_ranking",
    "sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping.rfe_initial_step_history",
    "sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping.rfe_elimination_threshold",
    "sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping.rfe_final_feature_count",
}


def test_rfe_state_bookkeeping_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_rfe_state_bookkeeping_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]
    for atom_entry in payload["atoms"].values():
        refs = atom_entry["references"]
        assert refs
        for ref in refs:
            ref_id = ref["ref_id"]
            assert ref_id in registry
            assert ref["match_metadata"]["confidence"] == "high"


def test_rfe_state_bookkeeping_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.feature_selection.rfe_state_bookkeeping.atoms")
    observed = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert observed == {
        "rfe_initial_support_mask",
        "rfe_initial_ranking",
        "rfe_initial_step_history",
        "rfe_elimination_threshold",
        "rfe_final_feature_count",
    }
