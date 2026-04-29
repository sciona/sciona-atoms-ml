from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "ml" / "sklearn" / "covariance" / "graphical_lasso_cv_bookkeeping" / "references.json"

EXPECTED_FQDNS = {
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_bookkeeping.graphical_lasso_cv_alpha_grid",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_bookkeeping.graphical_lasso_cv_mean_test_scores",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_bookkeeping.graphical_lasso_cv_best_index",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_bookkeeping.graphical_lasso_cv_refinement_bounds",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_bookkeeping.graphical_lasso_cv_refined_alpha_grid",
    "sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_bookkeeping.graphical_lasso_cv_results",
}


def test_graphical_lasso_cv_bookkeeping_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    observed = {key.split("@", 1)[0] for key in payload["atoms"]}
    assert observed == EXPECTED_FQDNS


def test_graphical_lasso_cv_bookkeeping_ref_ids_exist_and_have_metadata() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    for record in payload["atoms"].values():
        refs = record["references"]
        assert refs
        for ref in refs:
            assert ref["ref_id"]
            assert ref["match_metadata"]["match_type"]
            assert ref["match_metadata"]["confidence"]


def test_graphical_lasso_cv_bookkeeping_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_bookkeeping.atoms")
    expected_leaf_names = {fqdn.rsplit(".", 1)[-1] for fqdn in EXPECTED_FQDNS}
    assert len(expected_leaf_names) == len(EXPECTED_FQDNS)
