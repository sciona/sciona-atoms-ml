from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "covariance" / "graphical_lasso_fit_shell" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell.graphical_lasso_fit_use_precomputed_covariance",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell.graphical_lasso_fit_empirical_covariance",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell.graphical_lasso_fit_location",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell.graphical_lasso_fit_covariance",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell.graphical_lasso_fit_precision",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell.graphical_lasso_fit_costs",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell.graphical_lasso_fit_n_iter",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell.graphical_lasso_fit_return_self",
}


def test_graphical_lasso_fit_shell_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert atom_fqdns == EXPECTED_FQDNS
    assert "TO_UPDATE" not in REFERENCES_PATH.read_text(encoding="utf-8")


def test_graphical_lasso_fit_shell_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for entry in payload["atoms"].values():
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_graphical_lasso_fit_shell_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        assert fqdn.rsplit(".", 1)[-1] in registered
