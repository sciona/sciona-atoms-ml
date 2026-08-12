from __future__ import annotations

import numpy as np
import pandas as pd


def _mixed_table(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    numeric = rng.normal(size=rows)
    category = np.where(numeric > 0.4, "high", np.where(numeric < -0.4, "low", "mid"))
    noise = rng.normal(scale=0.35, size=rows)
    target = np.where(numeric + noise > 0.15, "yes", "no")
    return pd.DataFrame(
        {"numeric": numeric, "category": category, "with_missing": numeric.copy(), "target": target}
    ).assign(with_missing=lambda frame: frame.with_missing.mask(frame.index % 17 == 0))


def test_tabular_workflow_is_deterministic_and_improves_over_prior() -> None:
    from sciona.atoms.ml.tabular.supervised_classification import (
        fit_cross_validated_logistic,
        fit_one_hot_logistic,
        fit_prior_probability,
        predict_binary_probabilities,
        predict_prior_probabilities,
        stratified_tabular_split,
    )

    split = stratified_tabular_split(_mixed_table())
    repeated = stratified_tabular_split(_mixed_table())
    X_train, X_test, y_train, y_test = split
    assert np.array_equal(y_test, repeated[3])
    assert X_test.equals(repeated[1])

    prior, prior_targets = predict_prior_probabilities(
        fit_prior_probability(y_train), X_test, y_test
    )
    logistic, logistic_targets = predict_binary_probabilities(
        fit_one_hot_logistic(X_train, y_train), X_test, y_test
    )
    cv, cv_targets = predict_binary_probabilities(
        fit_cross_validated_logistic(X_train, y_train), X_test, y_test
    )

    def log_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped)))

    assert np.array_equal(prior_targets, logistic_targets)
    assert np.array_equal(logistic_targets, cv_targets)
    assert log_loss(logistic, logistic_targets) < log_loss(prior, prior_targets)
    assert log_loss(cv, cv_targets) <= log_loss(prior, prior_targets)
    assert np.all((0.0 <= cv) & (cv <= 1.0))


def test_tabular_workflow_rejects_nonbinary_targets() -> None:
    from sciona.atoms.ml.tabular.supervised_classification import stratified_tabular_split

    table = pd.DataFrame({"feature": [1, 2, 3], "target": ["a", "b", "c"]})
    try:
        stratified_tabular_split(table)
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("expected nonbinary targets to be rejected")
