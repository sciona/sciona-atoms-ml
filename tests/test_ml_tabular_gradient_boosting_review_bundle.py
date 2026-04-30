from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tabular_gradient_boosting_review_bundle_covers_atoms() -> None:
    bundle = json.loads((ROOT / "data/review_bundles/ml_tabular_gradient_boosting.review_bundle.json").read_text())
    rows = {row["atom_name"]: row for row in bundle["rows"]}
    expected = {
        "sciona.atoms.ml.tabular.gradient_boosting.group_aggregate",
        "sciona.atoms.ml.tabular.gradient_boosting.aggregate_child_table",
        "sciona.atoms.ml.tabular.gradient_boosting.temporal_difference",
        "sciona.atoms.ml.tabular.gradient_boosting.rolling_statistics",
        "sciona.atoms.ml.tabular.gradient_boosting.time_decay_aggregate",
        "sciona.atoms.ml.tabular.gradient_boosting.pairwise_products",
        "sciona.atoms.ml.tabular.gradient_boosting.pairwise_ratios",
        "sciona.atoms.ml.tabular.gradient_boosting.missing_indicator_and_impute",
        "sciona.atoms.ml.tabular.gradient_boosting.rank_transform",
        "sciona.atoms.ml.tabular.gradient_boosting.frequency_encode_fit",
        "sciona.atoms.ml.tabular.gradient_boosting.frequency_encode",
        "sciona.atoms.ml.tabular.gradient_boosting.target_encode",
        "sciona.atoms.ml.tabular.gradient_boosting.tweedie_gradient",
        "sciona.atoms.ml.tabular.gradient_boosting.log_cosh_gradient",
        "sciona.atoms.ml.tabular.gradient_boosting.null_importance_p_values",
        "sciona.atoms.ml.tabular.gradient_boosting.extract_pseudo_labels",
    }

    assert set(rows) == expected
    for row in rows.values():
        assert row["review_status"] == "reviewed"
        assert row["has_references"] is True
        assert row["references_status"] == "pass"

