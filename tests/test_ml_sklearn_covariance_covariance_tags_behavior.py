from __future__ import annotations

from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance, GraphicalLasso, GraphicalLassoCV, MinCovDet


def test_covariance_tags_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_tags import (
        covariance_estimator_type_tag,
        covariance_has_classifier_tags,
        covariance_target_required_tag,
        elliptic_envelope_estimator_type_tag,
    )

    assert callable(covariance_target_required_tag)
    assert callable(covariance_estimator_type_tag)
    assert callable(covariance_has_classifier_tags)
    assert callable(elliptic_envelope_estimator_type_tag)


def test_covariance_tags_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_tags import (
        covariance_estimator_type_tag,
        covariance_has_classifier_tags,
        covariance_target_required_tag,
        elliptic_envelope_estimator_type_tag,
    )

    assert covariance_target_required_tag(False) is False
    assert covariance_estimator_type_tag(None) is None
    assert covariance_has_classifier_tags(False) is False
    assert elliptic_envelope_estimator_type_tag(None) == "outlier_detector"


def test_covariance_tags_match_sklearn_estimators() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_tags import (
        covariance_estimator_type_tag,
        covariance_has_classifier_tags,
        covariance_target_required_tag,
        elliptic_envelope_estimator_type_tag,
    )

    for estimator in (EmpiricalCovariance(), GraphicalLasso(), GraphicalLassoCV(), MinCovDet()):
        tags = estimator.__sklearn_tags__()
        assert covariance_target_required_tag(tags.target_tags.required) is False
        assert covariance_estimator_type_tag(tags.estimator_type) is None
        assert covariance_has_classifier_tags(tags.classifier_tags is not None) is False

    elliptic_tags = EllipticEnvelope().__sklearn_tags__()
    assert covariance_target_required_tag(elliptic_tags.target_tags.required) is False
    assert elliptic_envelope_estimator_type_tag(elliptic_tags.estimator_type) == "outlier_detector"
    assert covariance_has_classifier_tags(elliptic_tags.classifier_tags is not None) is False


def test_covariance_tags_contracts() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_tags import (
        covariance_estimator_type_tag,
        covariance_has_classifier_tags,
        covariance_target_required_tag,
        elliptic_envelope_estimator_type_tag,
    )

    import pytest

    with pytest.raises(Exception):
        covariance_target_required_tag(1)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        covariance_estimator_type_tag(1)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        covariance_has_classifier_tags(1)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        elliptic_envelope_estimator_type_tag(1)  # type: ignore[arg-type]
