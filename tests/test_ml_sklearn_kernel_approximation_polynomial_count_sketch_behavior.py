from __future__ import annotations

import numpy as np
import pytest
from sklearn.kernel_approximation import PolynomialCountSketch as SklearnPolynomialCountSketch


def _data() -> np.ndarray:
    return np.array(
        [
            [0.5, 1.0, 2.0],
            [1.5, 2.0, 3.0],
            [2.5, 1.0, 0.5],
            [3.5, 4.0, 1.5],
        ],
        dtype=np.float64,
    )


def test_polynomial_count_sketch_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import (
        PolynomialCountSketchState,
        polynomial_count_sketch_fit,
        polynomial_count_sketch_transform,
    )

    assert PolynomialCountSketchState is not None
    assert callable(polynomial_count_sketch_fit)
    assert callable(polynomial_count_sketch_transform)


def test_polynomial_count_sketch_fit_and_transform_match_sklearn_without_coef0() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import (
        polynomial_count_sketch_fit,
        polynomial_count_sketch_transform,
    )

    X = _data()
    state = polynomial_count_sketch_fit(X, gamma=0.75, degree=3, coef0=0.0, n_components=5, random_state=11)
    expected = SklearnPolynomialCountSketch(
        gamma=0.75,
        degree=3,
        coef0=0.0,
        n_components=5,
        random_state=11,
    ).fit(X)

    assert np.array_equal(state.index_hash, expected.indexHash_)
    assert np.array_equal(state.bit_hash, expected.bitHash_)
    assert state.n_features_in == expected.n_features_in_
    assert np.allclose(polynomial_count_sketch_transform(X, state), expected.transform(X))


def test_polynomial_count_sketch_fit_and_transform_match_sklearn_with_coef0() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import (
        polynomial_count_sketch_fit,
        polynomial_count_sketch_transform,
    )

    X = _data()
    state = polynomial_count_sketch_fit(X, gamma=1.25, degree=2, coef0=1.5, n_components=7, random_state=19)
    expected = SklearnPolynomialCountSketch(
        gamma=1.25,
        degree=2,
        coef0=1.5,
        n_components=7,
        random_state=19,
    ).fit(X)

    assert np.array_equal(state.index_hash, expected.indexHash_)
    assert np.array_equal(state.bit_hash, expected.bitHash_)
    assert state.index_hash.shape == (2, X.shape[1] + 1)
    assert np.allclose(polynomial_count_sketch_transform(X, state), expected.transform(X))


def test_polynomial_count_sketch_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.kernel_approximation import (
        polynomial_count_sketch_fit,
        polynomial_count_sketch_transform,
    )

    X = _data()
    with pytest.raises(Exception):
        polynomial_count_sketch_fit(X, gamma=0.0)
    with pytest.raises(Exception):
        polynomial_count_sketch_fit(X, degree=0)
    with pytest.raises(Exception):
        polynomial_count_sketch_fit(X, coef0=-0.5)
    with pytest.raises(Exception):
        polynomial_count_sketch_fit(X, n_components=0)

    state = polynomial_count_sketch_fit(X, n_components=3, random_state=0)
    with pytest.raises(Exception):
        polynomial_count_sketch_transform(X[:, :2], state)
