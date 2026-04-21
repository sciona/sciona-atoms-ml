from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "manifold" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.manifold.lle_barycenter_weights",
    "sciona.atoms.ml.sklearn.manifold.lle_barycenter_graph",
    "sciona.atoms.ml.sklearn.manifold.lle_standard_reconstruction_matrix",
    "sciona.atoms.ml.sklearn.manifold.locally_linear_embedding",
    "sciona.atoms.ml.sklearn.manifold.locally_linear_embedding_fit",
    "sciona.atoms.ml.sklearn.manifold.locally_linear_embedding_transform",
}


def test_lle_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert EXPECTED_FQDNS.issubset(atom_fqdns)


def test_lle_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] not in EXPECTED_FQDNS:
            continue
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_lle_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.manifold.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        assert fqdn.rsplit(".", 1)[-1] in registered
