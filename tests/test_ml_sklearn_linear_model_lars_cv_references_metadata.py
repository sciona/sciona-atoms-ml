from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "linear_model" / "lars_cv" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.linear_model.lars_cv.lars_cv_residual_path",
    "sciona.atoms.ml.sklearn.linear_model.lars_cv.lars_cv_alpha_grid",
    "sciona.atoms.ml.sklearn.linear_model.lars_cv.lars_cv_interpolated_fold_mse",
    "sciona.atoms.ml.sklearn.linear_model.lars_cv.lars_cv_finite_row_mask",
    "sciona.atoms.ml.sklearn.linear_model.lars_cv.lars_cv_best_alpha",
}


def test_lars_cv_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_fqdns = {key.partition("@")[0] for key in payload["atoms"]}
    assert atom_fqdns == EXPECTED_FQDNS


def test_lars_cv_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry_ids = set(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["references"])
    for entry in payload["atoms"].values():
        assert entry["references"]
        for ref in entry["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]
            assert ref["match_metadata"]["notes"]


def test_lars_cv_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.linear_model.lars_cv.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        assert fqdn.rsplit(".", 1)[-1] in registered
