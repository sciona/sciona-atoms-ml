from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.gaussian_process._gpc import (
    GaussianProcessClassifier,
    _BinaryGaussianProcessClassifierLaplace,
)


def test_classification_tags_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_tags import (
        gpc_binary_has_classifier_tags,
        gpc_binary_target_required_tag,
        gpc_classifier_estimator_type_tag,
        gpc_classifier_has_classifier_tags,
        gpc_classifier_target_required_tag,
    )

    assert callable(gpc_binary_target_required_tag)
    assert callable(gpc_binary_has_classifier_tags)
    assert callable(gpc_classifier_estimator_type_tag)
    assert callable(gpc_classifier_target_required_tag)
    assert callable(gpc_classifier_has_classifier_tags)


def test_classification_tags_match_sklearn_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_tags import (
        gpc_binary_has_classifier_tags,
        gpc_binary_target_required_tag,
        gpc_classifier_estimator_type_tag,
        gpc_classifier_has_classifier_tags,
        gpc_classifier_target_required_tag,
    )

    binary_tags = _BinaryGaussianProcessClassifierLaplace().__sklearn_tags__()
    classifier_tags = GaussianProcessClassifier().__sklearn_tags__()

    assert gpc_binary_target_required_tag(binary_tags.target_tags.required) is binary_tags.target_tags.required
    assert gpc_binary_has_classifier_tags(binary_tags.classifier_tags is not None) is False
    assert gpc_classifier_estimator_type_tag(classifier_tags.estimator_type) == classifier_tags.estimator_type
    assert gpc_classifier_target_required_tag(False) is classifier_tags.target_tags.required
    assert gpc_classifier_has_classifier_tags(classifier_tags.classifier_tags is not None) is True


def test_classification_tag_overrides_ignore_parent_values() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_tags import (
        gpc_binary_has_classifier_tags,
        gpc_binary_target_required_tag,
        gpc_classifier_estimator_type_tag,
        gpc_classifier_has_classifier_tags,
        gpc_classifier_target_required_tag,
    )

    assert gpc_binary_target_required_tag(True) is False
    assert gpc_binary_has_classifier_tags(True) is False
    assert gpc_classifier_estimator_type_tag(None) == "classifier"
    assert gpc_classifier_target_required_tag(False) is True
    assert gpc_classifier_has_classifier_tags(False) is True


def test_classification_tags_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_tags import (
        gpc_binary_target_required_tag,
        gpc_classifier_estimator_type_tag,
        gpc_classifier_has_classifier_tags,
    )

    with pytest.raises(ViolationError):
        gpc_binary_target_required_tag("yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        gpc_classifier_estimator_type_tag(1)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        gpc_classifier_has_classifier_tags("no")  # type: ignore[arg-type]
