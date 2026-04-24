from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "cluster" / "bicluster_structure" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.cluster.bicluster_structure.cocluster_singular_vector_count",
    "sciona.atoms.ml.sklearn.cluster.bicluster_structure.cocluster_stacked_embedding",
    "sciona.atoms.ml.sklearn.cluster.bicluster_structure.cocluster_split_labels",
    "sciona.atoms.ml.sklearn.cluster.bicluster_structure.cocluster_indicator_matrix",
    "sciona.atoms.ml.sklearn.cluster.bicluster_structure.bicluster_effective_svd_dims",
    "sciona.atoms.ml.sklearn.cluster.bicluster_structure.bicluster_resolve_cluster_counts",
    "sciona.atoms.ml.sklearn.cluster.bicluster_structure.bicluster_indicator_grid",
}


def test_bicluster_structure_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_bicluster_structure_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"]

    for atom_key, atom_payload in payload["atoms"].items():
        assert atom_key.split("@", 1)[0] in EXPECTED_FQDNS
        refs = atom_payload["references"]
        assert refs
        for ref in refs:
            assert ref["ref_id"] in registry
            assert ref["match_metadata"]["confidence"] in {"high", "medium", "low"}
            assert isinstance(ref["match_metadata"]["notes"], str) and ref["match_metadata"]["notes"]


def test_bicluster_structure_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.cluster.bicluster_structure.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    expected_leaf_names = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert expected_leaf_names.issubset(registered)
