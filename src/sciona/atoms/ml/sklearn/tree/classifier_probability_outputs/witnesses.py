"""Ghost witnesses for sklearn tree classifier probability-output atoms."""

from __future__ import annotations


def witness_tree_predict_proba_single_output(
    probabilities: object, n_classes: object
) -> object:
    """Describe the `proba[:, : self.n_classes_]` shell in DecisionTreeClassifier.predict_proba."""
    return probabilities, n_classes


def witness_tree_predict_proba_multioutput(
    probabilities: object, n_classes: object
) -> object:
    """Describe the per-output slicing shell in DecisionTreeClassifier.predict_proba."""
    return probabilities, n_classes


def witness_tree_predict_log_proba_single_output(probabilities: object) -> object:
    """Describe the `np.log(proba)` shell in single-output DecisionTreeClassifier.predict_log_proba."""
    return probabilities


def witness_tree_predict_log_proba_multioutput(probabilities: object) -> object:
    """Describe the per-output `np.log(proba[k])` shell in multioutput DecisionTreeClassifier.predict_log_proba."""
    return probabilities
