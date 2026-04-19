from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import image as sklearn_image


def _dense(matrix: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def test_all_four_image_functions_import() -> None:
    from sciona.atoms.ml.sklearn.images import (
        extract_patches_2d,
        grid_to_graph,
        img_to_graph,
        reconstruct_from_patches_2d,
    )

    assert callable(extract_patches_2d)
    assert callable(reconstruct_from_patches_2d)
    assert callable(img_to_graph)
    assert callable(grid_to_graph)


def test_extract_patches_matches_sklearn_for_2d_and_color_images() -> None:
    from sciona.atoms.ml.sklearn.images import extract_patches_2d

    grayscale = np.arange(16, dtype=np.float64).reshape(4, 4)
    color = np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3)

    assert np.array_equal(
        extract_patches_2d(grayscale, (2, 3)),
        sklearn_image.extract_patches_2d(grayscale, (2, 3)),
    )
    assert np.array_equal(
        extract_patches_2d(color, (2, 2)),
        sklearn_image.extract_patches_2d(color, (2, 2)),
    )


def test_extract_patches_seeded_sampling_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.images import extract_patches_2d

    source = np.arange(25, dtype=np.float64).reshape(5, 5)
    result = extract_patches_2d(source, (2, 2), max_patches=4, random_state=11)
    expected = sklearn_image.extract_patches_2d(source, (2, 2), max_patches=4, random_state=11)

    assert np.array_equal(result, expected)


def test_reconstruct_from_patches_matches_sklearn_and_round_trips() -> None:
    from sciona.atoms.ml.sklearn.images import extract_patches_2d, reconstruct_from_patches_2d

    source = np.arange(16, dtype=np.float64).reshape(4, 4)
    patches = extract_patches_2d(source, (2, 2))

    result = reconstruct_from_patches_2d(patches, source.shape)
    expected = sklearn_image.reconstruct_from_patches_2d(patches, source.shape)

    assert np.allclose(result, expected)
    assert np.allclose(result, source)


def test_img_to_graph_matches_sklearn_sparse_and_dense_outputs() -> None:
    from sciona.atoms.ml.sklearn.images import img_to_graph

    img = np.array([[0.0, 2.0], [4.0, 7.0]], dtype=np.float64)
    mask = np.array([[True, False], [True, True]])

    result_sparse = img_to_graph(img, mask=mask)
    expected_sparse = sklearn_image.img_to_graph(img, mask=mask)
    assert sp.issparse(result_sparse)
    assert np.array_equal(_dense(result_sparse), _dense(expected_sparse))

    result_dense = img_to_graph(img, return_as=np.ndarray)
    expected_dense = sklearn_image.img_to_graph(img, return_as=np.ndarray)
    assert np.array_equal(result_dense, expected_dense)


def test_grid_to_graph_matches_sklearn_sparse_and_masked_outputs() -> None:
    from sciona.atoms.ml.sklearn.images import grid_to_graph

    mask = np.zeros((3, 3, 1), dtype=bool)
    mask[[0, 1, 2], [0, 1, 2], :] = True

    result_sparse = grid_to_graph(3, 3, 1)
    expected_sparse = sklearn_image.grid_to_graph(3, 3, 1)
    assert sp.issparse(result_sparse)
    assert np.array_equal(_dense(result_sparse), _dense(expected_sparse))

    result_masked = grid_to_graph(3, 3, 1, mask=mask)
    expected_masked = sklearn_image.grid_to_graph(3, 3, 1, mask=mask)
    assert np.array_equal(_dense(result_masked), _dense(expected_masked))


def test_graph_contracts_hold_for_small_3d_inputs() -> None:
    from sciona.atoms.ml.sklearn.images import grid_to_graph, img_to_graph

    volume = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    image_graph = img_to_graph(volume)
    grid_graph = grid_to_graph(2, 2, 2)

    assert image_graph.shape == (8, 8)
    assert grid_graph.shape == (8, 8)
    assert image_graph.nnz > 0
    assert grid_graph.nnz > 0
