from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = (
    ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "feature_selection" / "selectors" / "references.json"
)
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.feature_selection.selectors.feature_importances_transform",
    "sciona.atoms.ml.sklearn.feature_selection.selectors.rfe_elimination_step",
    "sciona.atoms.ml.sklearn.feature_selection.selectors.select_from_model_support_mask",
    "sciona.atoms.ml.sklearn.feature_selection.selectors.select_from_model_threshold",
    "sciona.atoms.ml.sklearn.feature_selection.selectors.sequential_best_feature",
    "sciona.atoms.ml.sklearn.feature_selection.selectors.sequential_candidate_masks",
}


def test_references_json_exists_and_has_selector_fqdns() -> None:
    assert REFERENCES_PATH.exists()
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_keys = set(payload["atoms"])
    assert len(atom_keys) == 6
    assert {key.partition("@")[0] for key in atom_keys} == EXPECTED_FQDNS


def test_each_selector_atom_has_nonempty_references() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        assert entry["references"], f"empty references for {key}"


def test_selector_ref_ids_exist_in_local_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    for key, entry in payload["atoms"].items():
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids, f"{ref['ref_id']} not in registry for {key}"


def test_each_selector_reference_has_match_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        for ref in entry["references"]:
            metadata = ref["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"]
            assert metadata["notes"]


def test_selector_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.feature_selection.selectors.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.ml.sklearn.feature_selection.selectors.")
        assert leaf in registered, f"{leaf} not in REGISTRY"
