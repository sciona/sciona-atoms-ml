from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "cluster" / "bicluster" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.cluster.bicluster.bicluster_scale_normalize",
    "sciona.atoms.ml.sklearn.cluster.bicluster.bicluster_bistochastic_normalize",
    "sciona.atoms.ml.sklearn.cluster.bicluster.bicluster_log_normalize",
}


def test_bicluster_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert atom_fqdns == EXPECTED_FQDNS
    assert "TO_UPDATE" not in REFERENCES_PATH.read_text(encoding="utf-8")


def test_bicluster_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for entry in payload["atoms"].values():
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_bicluster_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.cluster.bicluster.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    assert {
        "bicluster_scale_normalize",
        "bicluster_bistochastic_normalize",
        "bicluster_log_normalize",
    }.issubset(registered)
