"""Reusable, deterministic atoms for mixed-type binary classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sciona.ghost.registry import register_atom
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .witnesses import (
    witness_fit_cross_validated_logistic,
    witness_fit_one_hot_logistic,
    witness_fit_prior_probability,
    witness_predict_binary_probabilities,
    witness_predict_prior_probabilities,
    witness_stratified_tabular_split,
)

RANDOM_STATE = 1729
ATOM_PREFIX = "sciona.atoms.ml.tabular.supervised_classification."


def _binary_targets(values: Any) -> NDArray[np.int64]:
    series = pd.Series(values).reset_index(drop=True)
    if series.isna().any():
        raise ValueError("binary target contains missing values")
    unique = sorted(series.unique().tolist(), key=str)
    if len(unique) != 2:
        raise ValueError("supervised tabular classification requires exactly two target classes")
    return (series == unique[-1]).to_numpy(dtype=np.int64)


def _preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("tabular dataset has no feature columns")
    return ColumnTransformer(transformers=transformers)


@register_atom(
    witness_stratified_tabular_split,
    name=ATOM_PREFIX + "stratified_tabular_split",
)
def stratified_tabular_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, NDArray[np.int64], NDArray[np.int64]]:
    """Split a labeled table reproducibly, treating its final column as target."""
    if not isinstance(dataset, pd.DataFrame) or dataset.shape[1] < 2:
        raise ValueError("dataset must be a DataFrame with features and a final target column")
    features = dataset.iloc[:, :-1].copy()
    targets = _binary_targets(dataset.iloc[:, -1])
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=targets,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(y_test, dtype=np.int64),
    )


@register_atom(
    witness_fit_prior_probability,
    name=ATOM_PREFIX + "fit_prior_probability",
)
def fit_prior_probability(y_train: NDArray[np.int64]) -> float:
    """Fit the deterministic empirical-prior baseline for binary targets."""
    targets = _binary_targets(y_train)
    return float(np.mean(targets))


@register_atom(
    witness_predict_prior_probabilities,
    name=ATOM_PREFIX + "predict_prior_probabilities",
)
def predict_prior_probabilities(
    class_probability: float,
    X_test: pd.DataFrame,
    y_test: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Emit constant positive-class probabilities aligned with held-out labels."""
    if not 0.0 < float(class_probability) < 1.0:
        raise ValueError("class_probability must be strictly between zero and one")
    probabilities = np.full(len(X_test), float(class_probability), dtype=np.float64)
    targets = np.asarray(y_test, dtype=np.int64).reshape(-1)
    if probabilities.size != targets.size:
        raise ValueError("held-out features and targets must have equal length")
    return probabilities, targets


def _fit_logistic(
    X_train: pd.DataFrame,
    y_train: NDArray[np.int64],
    *,
    cross_validated: bool,
) -> Pipeline:
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a DataFrame")
    targets = np.asarray(y_train, dtype=np.int64).reshape(-1)
    if len(X_train) != targets.size or np.unique(targets).size != 2:
        raise ValueError("training features require aligned binary targets")
    estimator: Any
    if cross_validated:
        estimator = LogisticRegressionCV(
            Cs=np.logspace(-2, 2, 7),
            l1_ratios=(0.0,),
            cv=5,
            scoring="neg_log_loss",
            max_iter=1500,
            random_state=RANDOM_STATE,
            n_jobs=1,
            use_legacy_attributes=False,
        )
    else:
        estimator = LogisticRegression(
            C=1.0,
            max_iter=1500,
            random_state=RANDOM_STATE,
        )
    model = Pipeline(
        [
            ("preprocess", _preprocessor(X_train)),
            ("classifier", estimator),
        ]
    )
    model.fit(X_train, targets)
    return model


@register_atom(
    witness_fit_one_hot_logistic,
    name=ATOM_PREFIX + "fit_one_hot_logistic",
)
def fit_one_hot_logistic(
    X_train: pd.DataFrame,
    y_train: NDArray[np.int64],
) -> Pipeline:
    """Fit a deterministic one-hot encoded logistic classifier."""
    return _fit_logistic(X_train, y_train, cross_validated=False)


@register_atom(
    witness_fit_cross_validated_logistic,
    name=ATOM_PREFIX + "fit_cross_validated_logistic",
)
def fit_cross_validated_logistic(
    X_train: pd.DataFrame,
    y_train: NDArray[np.int64],
) -> Pipeline:
    """Select logistic regularization by stratified held-in log-loss."""
    return _fit_logistic(X_train, y_train, cross_validated=True)


@register_atom(
    witness_predict_binary_probabilities,
    name=ATOM_PREFIX + "predict_binary_probabilities",
)
def predict_binary_probabilities(
    model: Any,
    X_test: pd.DataFrame,
    y_test: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Emit positive-class probabilities and aligned held-out targets."""
    probabilities = np.asarray(model.predict_proba(X_test), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("binary classifier must emit a two-column probability matrix")
    targets = np.asarray(y_test, dtype=np.int64).reshape(-1)
    if probabilities.shape[0] != targets.size:
        raise ValueError("prediction and held-out target lengths differ")
    return probabilities[:, 1], targets
