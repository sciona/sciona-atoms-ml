from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "cluster" / "birch_subcluster_math" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED = {
    "sciona.atoms.ml.sklearn.cluster.birch_subcluster_math.birch_subcluster_singleton",
    "sciona.atoms.ml.sklearn.cluster.birch_subcluster_math.birch_subcluster_update",
    "sciona.atoms.ml.sklearn.cluster.birch_subcluster_math.birch_subcluster_squared_radius",
    "sciona.atoms.ml.sklearn.cluster.birch_subcluster_math.birch_subcluster_merge",
    "sciona.atoms.ml.sklearn.cluster.birch_subcluster_math.birch_subcluster_radius",
}


def test_birch_subcluster_math_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    found = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert found == EXPECTED


def test_birch_subcluster_math_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text())
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for atom in payload["atoms"].values():
        for reference in atom["references"]:
            assert reference["ref_id"] in registry_ids
            assert "match_metadata" in reference


def test_birch_subcluster_math_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.cluster.birch_subcluster_math.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    assert {
        "birch_subcluster_singleton",
        "birch_subcluster_update",
        "birch_subcluster_squared_radius",
        "birch_subcluster_merge",
        "birch_subcluster_radius",
    }.issubset(registered)
