from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "covariance" / "graphical_lasso" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso.graphical_lasso_offdiag_l1_penalty",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso.graphical_lasso_log_likelihood",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso.graphical_lasso_objective",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso.graphical_lasso_dual_gap",
}


def test_graphical_lasso_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert atom_fqdns == EXPECTED_FQDNS
    assert "TO_UPDATE" not in REFERENCES_PATH.read_text(encoding="utf-8")


def test_graphical_lasso_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for entry in payload["atoms"].values():
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_graphical_lasso_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.covariance.graphical_lasso.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    assert {
        "graphical_lasso_offdiag_l1_penalty",
        "graphical_lasso_log_likelihood",
        "graphical_lasso_objective",
        "graphical_lasso_dual_gap",
    }.issubset(registered)
