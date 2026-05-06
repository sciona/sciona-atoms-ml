from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = (
    ROOT
    / "data"
    / "review_bundles"
    / "ml_sklearn_linear_model_coordinate_descent_cv_target_guards.review_bundle.json"
)
CDG_PATH = (
    ROOT
    / "src"
    / "sciona"
    / "atoms"
    / "ml"
    / "sklearn"
    / "linear_model"
    / "coordinate_descent_cv_target_guards"
    / "cdg.json"
)

EXPECTED_ATOMS = {
    "sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards.cd_cv_reference_preserving_validation_branch",
    "sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards.cd_cv_non_multitask_guard_required",
    "sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards.cd_cv_non_multitask_message",
    "sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards.cd_cv_multitask_sparse_guard_required",
    "sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards.cd_cv_multitask_sparse_message",
    "sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards.cd_cv_multitask_monotask_guard_required",
    "sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards.cd_cv_multitask_monotask_message",
    "sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_target_guards.cd_cv_scalar_sample_weight_becomes_none",
}


def test_bundle_exists_and_has_coordinate_descent_cv_target_guards_atoms() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())
    assert (
        bundle["bundle_id"]
        == "sciona.atoms.review_bundle.ml.sklearn_linear_model_coordinate_descent_cv_target_guards.v1"
    )
    assert bundle["family_batch"] == "ml_sklearn_linear_model_coordinate_descent_cv_target_guards"
    assert bundle["review_status"] == "reviewed"
    assert bundle["review_semantic_verdict"] == "pass_with_limits"
    assert bundle["review_developer_semantic_verdict"] == "pass_with_limits"
    assert (
        bundle["review_record_path"]
        == "data/review_bundles/ml_sklearn_linear_model_coordinate_descent_cv_target_guards.review_bundle.json"
    )
    row_atoms = {row["atom_key"] for row in bundle["rows"]}
    assert row_atoms == EXPECTED_ATOMS


def test_bundle_rows_align_with_cdg_atoms() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())
    cdg = json.loads(CDG_PATH.read_text())
    atomic_names = {node["name"] for node in cdg["nodes"] if node["status"] == "atomic"}
    assert atomic_names == {atom.rsplit(".", 1)[-1] for atom in EXPECTED_ATOMS}
    for row in bundle["rows"]:
        assert row["overall_verdict"] == "pass"
        assert row["references_status"] == "pass"
        assert row["parity_test_status"] == "pass"
