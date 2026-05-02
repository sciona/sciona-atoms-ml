from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "gaussian_process" / "classification_log_marginal_likelihood_shell" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.gpc_log_marginal_likelihood_require_theta_for_gradient",
    "sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.gpc_log_marginal_likelihood_cached_result",
    "sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.gpc_log_marginal_likelihood_require_no_multiclass_gradient",
    "sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.gpc_log_marginal_likelihood_use_binary_branch",
    "sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.gpc_log_marginal_likelihood_use_shared_theta",
    "sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.gpc_log_marginal_likelihood_use_compound_theta",
    "sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.gpc_log_marginal_likelihood_theta_slice",
    "sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.gpc_log_marginal_likelihood_mean",
    "sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.gpc_log_marginal_likelihood_theta_shape_message",
}


def test_gpc_log_marginal_likelihood_shell_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert atom_fqdns == EXPECTED_FQDNS
    assert "TO_UPDATE" not in REFERENCES_PATH.read_text(encoding="utf-8")


def test_gpc_log_marginal_likelihood_shell_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for entry in payload["atoms"].values():
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_gpc_log_marginal_likelihood_shell_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.gaussian_process.classification_log_marginal_likelihood_shell.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        assert fqdn.rsplit(".", 1)[-1] in registered
