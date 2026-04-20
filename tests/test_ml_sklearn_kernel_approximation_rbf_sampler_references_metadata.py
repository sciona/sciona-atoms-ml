from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "kernel_approximation" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.kernel_approximation.rbf_sampler_fit",
    "sciona.atoms.ml.sklearn.kernel_approximation.rbf_sampler_transform",
}


def test_references_json_exists_and_has_rbf_sampler_fqdns() -> None:
    assert REFERENCES_PATH.exists()
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    assert {key.partition("@")[0] for key in payload["atoms"]} == EXPECTED_FQDNS


def test_rbf_sampler_atoms_have_nonempty_references() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for entry in payload["atoms"].values():
        assert entry["references"]


def test_rbf_sampler_ref_ids_exist_in_local_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    for entry in payload["atoms"].values():
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids


def test_rbf_sampler_reference_has_match_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for entry in payload["atoms"].values():
        for ref in entry["references"]:
            metadata = ref["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"]
            assert metadata["notes"]


def test_rbf_sampler_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.kernel_approximation.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.ml.sklearn.kernel_approximation.")
        assert leaf in registered
