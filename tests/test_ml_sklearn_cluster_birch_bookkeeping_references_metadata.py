from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "cluster" / "birch_bookkeeping" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.cluster.birch_bookkeeping.birch_first_call",
    "sciona.atoms.ml.sklearn.cluster.birch_bookkeeping.birch_copy_warning_required",
    "sciona.atoms.ml.sklearn.cluster.birch_bookkeeping.birch_compute_labels_required",
    "sciona.atoms.ml.sklearn.cluster.birch_bookkeeping.birch_not_enough_centroids",
    "sciona.atoms.ml.sklearn.cluster.birch_bookkeeping.birch_identity_subcluster_labels",
    "sciona.atoms.ml.sklearn.cluster.birch_bookkeeping.birch_leaf_centers",
    "sciona.atoms.ml.sklearn.cluster.birch_bookkeeping.birch_n_features_out",
}


def test_birch_bookkeeping_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert atom_fqdns == EXPECTED_FQDNS
    assert "TO_UPDATE" not in REFERENCES_PATH.read_text(encoding="utf-8")


def test_birch_bookkeeping_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for entry in payload["atoms"].values():
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_birch_bookkeeping_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.cluster.birch_bookkeeping.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    assert {
        "birch_first_call",
        "birch_copy_warning_required",
        "birch_compute_labels_required",
        "birch_not_enough_centroids",
        "birch_identity_subcluster_labels",
        "birch_leaf_centers",
        "birch_n_features_out",
    }.issubset(registered)
