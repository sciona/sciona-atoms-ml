"""Estimator-independent sklearn voting aggregation atoms."""

from .atoms import (
    voting_classifier_hard_labels,
    voting_classifier_soft_probabilities,
    voting_regressor_average,
)

__all__ = [
    "voting_classifier_hard_labels",
    "voting_classifier_soft_probabilities",
    "voting_regressor_average",
]
