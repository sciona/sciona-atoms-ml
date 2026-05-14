from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.linear_model import HuberRegressor


def test_huber_tags_super_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_tags_super_callback_shell import (
        huber_tags_return,
        huber_tags_sparse_input_value,
        huber_tags_super_result,
    )

    assert callable(huber_tags_super_result)
    assert callable(huber_tags_sparse_input_value)
    assert callable(huber_tags_return)


def test_huber_tags_super_callback_shell_matches_sklearn_sparse_tag() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_tags_super_callback_shell import (
        huber_tags_sparse_input_value,
    )

    tags = HuberRegressor().__sklearn_tags__()

    assert huber_tags_sparse_input_value(tags) is tags.input_tags.sparse


def test_huber_tags_super_callback_shell_preserves_tags_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_tags_super_callback_shell import (
        huber_tags_return,
        huber_tags_super_result,
    )

    tags = object()

    assert huber_tags_super_result(tags) is tags
    assert huber_tags_return(tags) is tags


def test_huber_tags_super_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_tags_super_callback_shell import (
        huber_tags_return,
        huber_tags_sparse_input_value,
        huber_tags_super_result,
    )

    with pytest.raises(ViolationError):
        huber_tags_super_result(None)

    with pytest.raises(ViolationError):
        huber_tags_sparse_input_value(None)

    with pytest.raises(ViolationError):
        huber_tags_return(None)
