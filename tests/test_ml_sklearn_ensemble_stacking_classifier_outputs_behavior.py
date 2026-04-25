from __future__ import annotations

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


def test_stacking_classifier_outputs_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_classifier_outputs import (
        stacking_classifier_labels_from_encoded,
        stacking_classifier_multilabel_labels_from_encoded,
        stacking_classifier_probability_matrix_from_blocks,
    )

    assert callable(stacking_classifier_labels_from_encoded)
    assert callable(stacking_classifier_multilabel_labels_from_encoded)
    assert callable(stacking_classifier_probability_matrix_from_blocks)


def test_stacking_classifier_labels_from_encoded_matches_sklearn_predict() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_classifier_outputs import stacking_classifier_labels_from_encoded

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )
    y = np.asarray(["left", "left", "right", "right", "left", "right"], dtype=object)
    clf = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=5, random_state=0))],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        cv=2,
    ).fit(X, y)

    encoded = np.asarray(clf.final_estimator_.predict(clf.transform(X)), dtype=np.int64)
    result = stacking_classifier_labels_from_encoded(encoded, np.asarray(clf.classes_, dtype=object))

    assert np.array_equal(result, clf.predict(X))


def test_stacking_classifier_multilabel_outputs_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_classifier_outputs import (
        stacking_classifier_multilabel_labels_from_encoded,
        stacking_classifier_probability_matrix_from_blocks,
    )

    X, Y = make_multilabel_classification(
        n_samples=60,
        n_features=8,
        n_classes=3,
        n_labels=1,
        random_state=0,
    )
    clf = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=5, random_state=0))],
        final_estimator=MultiOutputClassifier(LogisticRegression(max_iter=1000)),
        stack_method="predict_proba",
        cv=2,
    ).fit(X, Y)

    transformed = clf.transform(X)
    encoded = np.asarray(clf.final_estimator_.predict(transformed), dtype=np.int64)
    probability_blocks = tuple(
        np.asarray(block, dtype=np.float64)
        for block in clf.final_estimator_.predict_proba(transformed)
    )
    classes_blocks = tuple(np.asarray(block, dtype=object) for block in clf.classes_)

    label_result = stacking_classifier_multilabel_labels_from_encoded(encoded, classes_blocks)
    probability_result = stacking_classifier_probability_matrix_from_blocks(probability_blocks)

    assert np.array_equal(label_result, clf.predict(X))
    assert np.allclose(probability_result, np.asarray(clf.predict_proba(X), dtype=np.float64))


def test_contracts_reject_invalid_stacking_classifier_output_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_classifier_outputs import (
        stacking_classifier_labels_from_encoded,
        stacking_classifier_multilabel_labels_from_encoded,
        stacking_classifier_probability_matrix_from_blocks,
    )

    try:
        stacking_classifier_labels_from_encoded(
            np.asarray([0, 2], dtype=np.int64),
            np.asarray(["a", "b"], dtype=object),
        )
    except Exception as exc:
        assert exc.__class__.__name__ in {"ViolationError", "PreconditionError"}
    else:
        raise AssertionError("expected contract failure for out-of-range encoded labels")

    try:
        stacking_classifier_multilabel_labels_from_encoded(
            np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            (np.asarray(["a", "b"], dtype=object),),
        )
    except Exception as exc:
        assert exc.__class__.__name__ in {"ViolationError", "PreconditionError"}
    else:
        raise AssertionError("expected contract failure for mismatched class blocks")

    try:
        stacking_classifier_probability_matrix_from_blocks(
            (
                np.asarray([[0.5, 0.5]], dtype=np.float64),
                np.asarray([[0.5, 0.4]], dtype=np.float64),
            )
        )
    except Exception as exc:
        assert exc.__class__.__name__ in {"ViolationError", "PreconditionError"}
    else:
        raise AssertionError("expected contract failure for non-normalized probability blocks")
