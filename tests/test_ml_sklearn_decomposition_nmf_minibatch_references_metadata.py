from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "decomposition" / "nmf_minibatch" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.decomposition.nmf_minibatch.nmf_minibatch_batch_size",
    "sciona.atoms.ml.sklearn.decomposition.nmf_minibatch.nmf_minibatch_rho",
    "sciona.atoms.ml.sklearn.decomposition.nmf_minibatch.nmf_minibatch_mm_gamma",
    "sciona.atoms.ml.sklearn.decomposition.nmf_minibatch.nmf_minibatch_transform_max_iter",
    "sciona.atoms.ml.sklearn.decomposition.nmf_minibatch.nmf_minibatch_ewa_cost",
    "sciona.atoms.ml.sklearn.decomposition.nmf_minibatch.nmf_minibatch_h_change_converged",
    "sciona.atoms.ml.sklearn.decomposition.nmf_minibatch.nmf_minibatch_improvement_state",
}


def test_nmf_minibatch_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    fqdns = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert fqdns == EXPECTED_ATOMS


def test_nmf_minibatch_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])

    for fqdn, entry in payload["atoms"].items():
        assert fqdn.split("@", 1)[0] in EXPECTED_ATOMS
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            metadata = ref["match_metadata"]
            assert metadata["match_type"] == "manual"
            assert metadata["confidence"] in {"high", "medium", "low"}
            assert isinstance(metadata["notes"], str) and metadata["notes"].strip()


def test_nmf_minibatch_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.decomposition.nmf_minibatch.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    expected_leaf_names = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_ATOMS}
    missing = expected_leaf_names - registered
    assert not missing
