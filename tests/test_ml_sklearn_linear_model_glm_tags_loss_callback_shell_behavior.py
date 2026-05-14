from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor


def test_glm_tags_loss_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_tags_loss_callback_shell import (
        glm_base_default_loss_name,
        glm_tags_exception_fallback,
        glm_tags_loss_callback_result,
        glm_tags_positive_only_from_negative_range,
        glm_tags_return,
        glm_tags_sparse_input_value,
        glm_tags_super_result,
    )

    assert callable(glm_tags_super_result)
    assert callable(glm_tags_sparse_input_value)
    assert callable(glm_tags_loss_callback_result)
    assert callable(glm_tags_positive_only_from_negative_range)
    assert callable(glm_tags_exception_fallback)
    assert callable(glm_tags_return)
    assert callable(glm_base_default_loss_name)


@pytest.mark.parametrize(
    "estimator",
    [
        PoissonRegressor(),
        GammaRegressor(),
        TweedieRegressor(power=1.5),
        TweedieRegressor(power=0.0),
    ],
)
def test_glm_tags_loss_callback_shell_matches_sklearn_tag_values(estimator: object) -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_tags_loss_callback_shell import (
        glm_tags_loss_callback_result,
        glm_tags_positive_only_from_negative_range,
        glm_tags_sparse_input_value,
    )

    tags = estimator.__sklearn_tags__()
    base_loss = estimator._get_loss()
    in_negative_range = base_loss.in_y_true_range(-1.0)

    assert glm_tags_sparse_input_value(tags) is tags.input_tags.sparse
    assert glm_tags_loss_callback_result(base_loss) is base_loss
    assert glm_tags_positive_only_from_negative_range(in_negative_range) is tags.target_tags.positive_only


def test_glm_tags_loss_callback_shell_preserves_identities() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_tags_loss_callback_shell import (
        glm_tags_exception_fallback,
        glm_tags_loss_callback_result,
        glm_tags_return,
        glm_tags_super_result,
    )

    tags = object()
    base_loss = object()

    assert glm_tags_super_result(tags) is tags
    assert glm_tags_loss_callback_result(base_loss) is base_loss
    assert glm_tags_exception_fallback(tags, "ValueError") is tags
    assert glm_tags_exception_fallback(tags, "AttributeError") is tags
    assert glm_tags_exception_fallback(tags, "TypeError") is tags
    assert glm_tags_return(tags) is tags


def test_glm_base_default_loss_name_matches_sklearn_source() -> None:
    from sklearn.linear_model._glm.glm import _GeneralizedLinearRegressor

    from sciona.atoms.ml.sklearn.linear_model.glm_tags_loss_callback_shell import (
        glm_base_default_loss_name,
    )

    loss_name = type(_GeneralizedLinearRegressor()._get_loss()).__name__

    assert loss_name == "HalfSquaredError"
    assert glm_base_default_loss_name(loss_name) == loss_name


def test_glm_tags_loss_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.glm_tags_loss_callback_shell import (
        glm_base_default_loss_name,
        glm_tags_exception_fallback,
        glm_tags_loss_callback_result,
        glm_tags_positive_only_from_negative_range,
        glm_tags_return,
        glm_tags_sparse_input_value,
        glm_tags_super_result,
    )

    with pytest.raises(ViolationError):
        glm_tags_super_result(None)

    with pytest.raises(ViolationError):
        glm_tags_sparse_input_value(None)

    with pytest.raises(ViolationError):
        glm_tags_loss_callback_result(None)

    with pytest.raises(ViolationError):
        glm_tags_positive_only_from_negative_range(1)

    with pytest.raises(ViolationError):
        glm_tags_exception_fallback(object(), "RuntimeError")

    with pytest.raises(ViolationError):
        glm_tags_return(None)

    with pytest.raises(ViolationError):
        glm_base_default_loss_name("HalfPoissonLoss")
