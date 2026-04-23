from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "gaussian_process" / "classification" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.gaussian_process.classification.gp_classifier_laplace_log_marginal_likelihood",
    "sciona.atoms.ml.sklearn.gaussian_process.classification.gp_classifier_laplace_newton_step",
    "sciona.atoms.ml.sklearn.gaussian_process.classification.gp_classifier_posterior_cross_solve",
    "sciona.atoms.ml.sklearn.gaussian_process.classification.gp_classifier_posterior_mean",
    "sciona.atoms.ml.sklearn.gaussian_process.classification.gp_classifier_posterior_variance",
    "sciona.atoms.ml.sklearn.gaussian_process.classification.gp_classifier_predictive_proba",
    "sciona.atoms.ml.sklearn.gaussian_process.classification.gp_classifier_predictive_probability",
}


def test_gp_classifier_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_gp_classifier_ref_ids_exist_and_have_metadata() -> None:
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


def test_gp_classifier_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.gaussian_process.classification.atoms")
