from __future__ import annotations

import numpy as np
import pytest
from sklearn.kernel_approximation import Nystroem as SklearnNystroem


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [3.0, 4.0, 1.5],
            [4.0, 2.0, 2.5],
        ],
        dtype=np.float64,
    )


def test_nystroem_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import NystroemState, nystroem_fit, nystroem_transform

    assert NystroemState is not None
    assert callable(nystroem_fit)
    assert callable(nystroem_transform)


def test_nystroem_fit_and_transform_match_sklearn_rbf() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import nystroem_fit, nystroem_transform

    X = _data()
    state = nystroem_fit(X, kernel="rbf", gamma=0.25, n_components=3, random_state=17)
    expected = SklearnNystroem(kernel="rbf", gamma=0.25, n_components=3, random_state=17).fit(X)

    assert np.array_equal(state.component_indices, expected.component_indices_)
    assert np.allclose(state.components, expected.components_)
    assert np.allclose(state.normalization, expected.normalization_)
    assert state.n_components == expected.components_.shape[0]
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(nystroem_transform(X, state), expected.transform(X))


def test_nystroem_fit_and_transform_match_sklearn_polynomial_kernel_params() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import nystroem_fit, nystroem_transform

    X = _data()
    state = nystroem_fit(
        X,
        kernel="polynomial",
        gamma=0.5,
        degree=3.0,
        coef0=1.25,
        n_components=4,
        random_state=5,
    )
    expected = SklearnNystroem(
        kernel="polynomial",
        gamma=0.5,
        degree=3.0,
        coef0=1.25,
        n_components=4,
        random_state=5,
    ).fit(X)

    assert state.kernel_params == {"gamma": 0.5, "degree": 3.0, "coef0": 1.25}
    assert np.allclose(state.normalization, expected.normalization_)
    assert np.allclose(nystroem_transform(X[:3], state), expected.transform(X[:3]))


def test_nystroem_caps_components_to_sample_count_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import nystroem_fit, nystroem_transform

    X = _data()[:3]
    state = nystroem_fit(X, kernel="linear", n_components=10, random_state=3)
    with pytest.warns(UserWarning):
        expected = SklearnNystroem(kernel="linear", n_components=10, random_state=3).fit(X)

    assert state.n_components == X.shape[0]
    assert np.array_equal(state.component_indices, expected.component_indices_)
    assert np.allclose(nystroem_transform(X, state), expected.transform(X))


def test_nystroem_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import nystroem_fit, nystroem_transform

    X = _data()
    with pytest.raises(Exception):
        nystroem_fit(X, kernel="precomputed")
    with pytest.raises(Exception):
        nystroem_fit(X, gamma=-0.1)
    with pytest.raises(Exception):
        nystroem_fit(X, degree=0.5)
    with pytest.raises(Exception):
        nystroem_fit(X, n_components=0)
    with pytest.raises(Exception):
        nystroem_fit(np.array([[1.0, -1.0]], dtype=np.float64), kernel="chi2")

    state = nystroem_fit(X, n_components=3, random_state=0)
    with pytest.raises(Exception):
        nystroem_transform(np.ones((2, 2), dtype=np.float64), state)
