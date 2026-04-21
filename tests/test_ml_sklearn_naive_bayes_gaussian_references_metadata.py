from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "naive_bayes" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.naive_bayes.gaussian_nb_update_mean_variance",
    "sciona.atoms.ml.sklearn.naive_bayes.gaussian_nb_fit",
    "sciona.atoms.ml.sklearn.naive_bayes.gaussian_nb_joint_log_likelihood",
    "sciona.atoms.ml.sklearn.naive_bayes.gaussian_nb_predict_log_proba",
    "sciona.atoms.ml.sklearn.naive_bayes.gaussian_nb_predict_proba",
    "sciona.atoms.ml.sklearn.naive_bayes.gaussian_nb_predict",
}


def test_gaussian_nb_references_json_exists_and_has_fqdns() -> None:
    assert REFERENCES_PATH.exists()
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    assert EXPECTED_FQDNS == {key.partition("@")[0] for key in payload["atoms"]}


def test_gaussian_nb_atoms_have_nonempty_references() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for entry in payload["atoms"].values():
        assert entry["references"]


def test_gaussian_nb_ref_ids_exist_in_local_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    for entry in payload["atoms"].values():
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids


def test_gaussian_nb_reference_has_match_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for entry in payload["atoms"].values():
        for ref in entry["references"]:
            metadata = ref["match_metadata"]
            assert metadata["match_type"]
            assert metadata["confidence"]
            assert metadata["notes"]


def test_gaussian_nb_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.naive_bayes.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.ml.sklearn.naive_bayes.")
        assert leaf in registered
