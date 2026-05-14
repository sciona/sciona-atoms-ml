from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.linear_model import SGDClassifier, SGDRegressor


def test_sgd_tags_super_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_tags_super_callback_shell import (
        sgd_tags_return,
        sgd_tags_sparse_input_value,
        sgd_tags_super_result,
    )

    assert callable(sgd_tags_super_result)
    assert callable(sgd_tags_sparse_input_value)
    assert callable(sgd_tags_return)


@pytest.mark.parametrize("estimator", [SGDClassifier(), SGDRegressor()])
def test_sgd_tags_super_callback_shell_matches_sklearn_sparse_tag(estimator: object) -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_tags_super_callback_shell import (
        sgd_tags_sparse_input_value,
    )

    tags = estimator.__sklearn_tags__()

    assert sgd_tags_sparse_input_value(tags) is tags.input_tags.sparse


def test_sgd_tags_super_callback_shell_preserves_tags_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_tags_super_callback_shell import (
        sgd_tags_return,
        sgd_tags_super_result,
    )

    tags = object()

    assert sgd_tags_super_result(tags) is tags
    assert sgd_tags_return(tags) is tags


def test_sgd_tags_super_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_tags_super_callback_shell import (
        sgd_tags_return,
        sgd_tags_sparse_input_value,
        sgd_tags_super_result,
    )

    with pytest.raises(ViolationError):
        sgd_tags_super_result(None)

    with pytest.raises(ViolationError):
        sgd_tags_sparse_input_value(None)

    with pytest.raises(ViolationError):
        sgd_tags_return(None)
