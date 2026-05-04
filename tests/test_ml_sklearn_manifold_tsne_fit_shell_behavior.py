from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.manifold import TSNE


def test_tsne_fit_shell_atom_import() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_shell import tsne_fit_return_self

    assert callable(tsne_fit_return_self)


def test_tsne_fit_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_shell import tsne_fit_return_self

    estimator = TSNE()
    X = np.array([[0.0], [1.0]], dtype=np.float64)

    with patch.object(TSNE, "fit_transform", autospec=True, return_value=np.zeros((2, 2), dtype=np.float64)) as mocked_fit_transform:
        result = estimator.fit(X)

    assert result is estimator
    mocked_fit_transform.assert_called_once_with(estimator, X)
    assert tsne_fit_return_self("TSNE") == "TSNE"


def test_tsne_fit_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_shell import tsne_fit_return_self

    with pytest.raises(Exception):
        tsne_fit_return_self("")
