from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "inspection" / "partial_dependence_result_packaging" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_result_packaging.partial_dependence_grid_value_lengths",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_result_packaging.partial_dependence_grid_shaped_averages",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_result_packaging.partial_dependence_grid_shaped_individual",
    "sciona.atoms.ml.sklearn.inspection.partial_dependence_result_packaging.partial_dependence_result_bunch",
}


def test_partial_dependence_result_packaging_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert atom_fqdns == EXPECTED_FQDNS
    assert "TO_UPDATE" not in REFERENCES_PATH.read_text(encoding="utf-8")


def test_partial_dependence_result_packaging_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for entry in payload["atoms"].values():
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_partial_dependence_result_packaging_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.inspection.partial_dependence_result_packaging.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    assert {
        "partial_dependence_grid_value_lengths",
        "partial_dependence_grid_shaped_averages",
        "partial_dependence_grid_shaped_individual",
        "partial_dependence_result_bunch",
    }.issubset(registered)
