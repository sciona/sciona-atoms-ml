from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "model_selection" / "recommendations" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.model_selection.recommendations.recommend_regularization",
    "sciona.atoms.ml.model_selection.recommendations.recommend_loss_function",
    "sciona.atoms.ml.model_selection.recommendations.recommend_linear_model",
    "sciona.atoms.ml.model_selection.recommendations.recommend_tree_ensemble",
    "sciona.atoms.ml.model_selection.recommendations.recommend_preprocessing",
    "sciona.atoms.ml.model_selection.recommendations.recommend_dimensionality_reduction",
    "sciona.atoms.ml.model_selection.recommendations.recommend_hyperparameter_ranges",
    "sciona.atoms.ml.model_selection.recommendations.recommend_cv_strategy",
}


def test_references_json_exists_and_has_recommendation_fqdns() -> None:
    assert REFERENCES_PATH.exists()
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert EXPECTED_FQDNS.issubset(atom_fqdns)


def test_recommendation_atoms_have_nonempty_references() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            assert entry["references"], f"empty references for {key}"


def test_recommendation_ref_ids_exist_in_local_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            for ref in entry["references"]:
                assert ref["ref_id"] in registry_ids, f"{ref['ref_id']} not in registry for {key}"


def test_recommendation_reference_has_match_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for key, entry in payload["atoms"].items():
        if key.partition("@")[0] in EXPECTED_FQDNS:
            for ref in entry["references"]:
                metadata = ref["match_metadata"]
                assert metadata["match_type"]
                assert metadata["confidence"]
                assert metadata["notes"]


def test_recommendation_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.model_selection.recommendations.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.ml.model_selection.recommendations.")
        assert leaf in registered
