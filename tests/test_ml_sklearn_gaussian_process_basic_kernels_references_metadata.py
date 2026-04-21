from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "gaussian_process" / "kernels" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.gaussian_process.kernels.constant_kernel",
    "sciona.atoms.ml.sklearn.gaussian_process.kernels.constant_kernel_diag",
    "sciona.atoms.ml.sklearn.gaussian_process.kernels.white_kernel",
    "sciona.atoms.ml.sklearn.gaussian_process.kernels.white_kernel_diag",
    "sciona.atoms.ml.sklearn.gaussian_process.kernels.dot_product_kernel",
    "sciona.atoms.ml.sklearn.gaussian_process.kernels.dot_product_kernel_diag",
    "sciona.atoms.ml.sklearn.gaussian_process.kernels.rbf_kernel_matrix",
    "sciona.atoms.ml.sklearn.gaussian_process.kernels.rbf_kernel_diag",
}


def test_references_json_exists_and_has_basic_kernel_fqdns() -> None:
    assert REFERENCES_PATH.exists()
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    assert EXPECTED_FQDNS.issubset({key.partition("@")[0] for key in payload["atoms"]})


def test_basic_kernel_atoms_have_nonempty_references() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            assert entry["references"]


def test_basic_kernel_ref_ids_exist_in_local_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            for ref in entry["references"]:
                assert ref["ref_id"] in registry_ids


def test_basic_kernel_reference_has_match_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            for ref in entry["references"]:
                metadata = ref["match_metadata"]
                assert metadata["match_type"]
                assert metadata["confidence"]
                assert metadata["notes"]


def test_basic_kernel_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.ml.sklearn.gaussian_process.kernels.")
        assert leaf in registered
