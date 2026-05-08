"""Image patch and grid graph atoms adapted from scikit-learn.

The implementations mirror the public function behavior of
``sklearn.feature_extraction.image`` for the pure function targets in the
sklearn 1.8.0 image module. The source project is BSD-3-Clause licensed.
"""

from __future__ import annotations

from itertools import product
from numbers import Integral, Real

import icontract
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_extract_patches_2d,
    witness_grid_to_graph,
    witness_img_to_graph,
    witness_patch_extractor_transform,
    witness_reconstruct_from_patches_2d,
)

PatchSize = tuple[int, int]
ImageSize = tuple[int, ...]
RandomStateArg = int | np.random.RandomState | None
SparseReturn = type[sp.spmatrix] | type[np.ndarray]
GraphResult = sp.spmatrix | NDArray[np.float64]

def _valid_max_patches(max_patches: int | float | None) -> bool:
    if max_patches is None:
        return True
    if isinstance(max_patches, Integral):
        return int(max_patches) >= 1
    if isinstance(max_patches, Real):
        return 0.0 < float(max_patches) < 1.0
    return False

def _valid_patch_size_for_images(X: NDArray[np.float64], patch_size: PatchSize | None) -> bool:
    values = np.asarray(X)
    if values.ndim not in {3, 4}:
        return False
    image_height, image_width = values.shape[1:3]
    if patch_size is None:
        patch_height, patch_width = image_height // 10, image_width // 10
    else:
        if len(patch_size) != 2:
            return False
        patch_height, patch_width = patch_size
    return bool(0 < patch_height <= image_height and 0 < patch_width <= image_width)

def _compute_n_patches(
    image_height: int,
    image_width: int,
    patch_height: int,
    patch_width: int,
    max_patches: int | float | None = None,
) -> int:
    n_h = image_height - patch_height + 1
    n_w = image_width - patch_width + 1
    all_patches = n_h * n_w

    if max_patches:
        if isinstance(max_patches, Integral) and max_patches < all_patches:
            return int(max_patches)
        if isinstance(max_patches, Integral) and max_patches >= all_patches:
            return int(all_patches)
        if isinstance(max_patches, Real) and 0.0 < float(max_patches) < 1.0:
            return int(float(max_patches) * all_patches)
        raise ValueError(f"Invalid value for max_patches: {max_patches!r}")
    return int(all_patches)

def _extract_patches(
    array: NDArray[np.float64],
    patch_shape: tuple[int, ...],
    extraction_step: int = 1,
) -> NDArray[np.float64]:
    array_ndim = array.ndim
    step = tuple([extraction_step] * array_ndim)
    patch_strides = array.strides
    slices = tuple(slice(None, None, item) for item in step)
    indexing_strides = array[slices].strides
    patch_indices_shape = (
        (np.array(array.shape) - np.array(patch_shape)) // np.array(step)
    ) + 1
    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    return as_strided(array, shape=shape, strides=strides)

def _make_edges_3d(n_x: int, n_y: int, n_z: int = 1) -> NDArray[np.int64]:
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    return np.hstack((edges_deep, edges_right, edges_down)).astype(np.int64)

def _compute_gradient_3d(
    edges: NDArray[np.int64],
    image: NDArray[np.float64],
) -> NDArray[np.float64]:
    _, n_y, n_z = image.shape
    gradient = np.abs(
        image[
            edges[0] // (n_y * n_z),
            (edges[0] % (n_y * n_z)) // n_z,
            (edges[0] % (n_y * n_z)) % n_z,
        ]
        - image[
            edges[1] // (n_y * n_z),
            (edges[1] % (n_y * n_z)) // n_z,
            (edges[1] % (n_y * n_z)) % n_z,
        ]
    )
    return np.asarray(gradient)

def _mask_edges(
    mask: NDArray[np.bool_],
    edges: NDArray[np.int64],
) -> NDArray[np.int64]:
    indices = np.arange(mask.size)
    active_indices = indices[mask.ravel()]
    edge_mask = np.logical_and(
        np.isin(edges[0], active_indices),
        np.isin(edges[1], active_indices),
    )
    masked_edges = edges[:, edge_mask]
    max_value = int(masked_edges.max()) if len(masked_edges.ravel()) else 0
    order = np.searchsorted(np.flatnonzero(mask), np.arange(max_value + 1))
    return np.asarray(order[masked_edges], dtype=np.int64)

def _mask_edges_and_weights(
    mask: NDArray[np.bool_],
    edges: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    indices = np.arange(mask.size)
    active_indices = indices[mask.ravel()]
    edge_mask = np.logical_and(
        np.isin(edges[0], active_indices),
        np.isin(edges[1], active_indices),
    )
    masked_edges = edges[:, edge_mask]
    masked_weights = weights[edge_mask]
    max_value = int(masked_edges.max()) if len(masked_edges.ravel()) else 0
    order = np.searchsorted(np.flatnonzero(mask), np.arange(max_value + 1))
    return np.asarray(order[masked_edges], dtype=np.int64), np.asarray(masked_weights)

def _coo_or_dense(
    graph: sp.coo_matrix,
    return_as: SparseReturn,
) -> GraphResult:
    if return_as is np.ndarray:
        return np.asarray(graph.toarray())
    return return_as(graph)

def _to_graph(
    n_x: int,
    n_y: int,
    n_z: int,
    *,
    mask: NDArray[np.bool_] | None = None,
    image: NDArray[np.float64] | None = None,
    return_as: SparseReturn = sp.coo_matrix,
    dtype: npt.DTypeLike | None = None,
) -> GraphResult:
    edges = _make_edges_3d(n_x, n_y, n_z)
    graph_dtype = image.dtype if dtype is None and image is not None else dtype
    if graph_dtype is None:
        graph_dtype = int

    if image is not None:
        image_3d = np.atleast_3d(image)
        weights = _compute_gradient_3d(edges, image_3d)
        if mask is not None:
            edges, weights = _mask_edges_and_weights(mask, edges, weights)
            diagonal = image_3d.squeeze()[mask]
        else:
            diagonal = image_3d.ravel()
        n_voxels = int(diagonal.size)
    else:
        if mask is not None:
            bool_mask = mask.astype(dtype=bool, copy=False)
            edges = _mask_edges(bool_mask, edges)
            n_voxels = int(np.sum(bool_mask))
        else:
            n_voxels = int(n_x * n_y * n_z)
        weights = np.ones(edges.shape[1], dtype=graph_dtype)
        diagonal = np.ones(n_voxels, dtype=graph_dtype)

    diagonal_index = np.arange(n_voxels)
    row_index = np.hstack((edges[0], edges[1]))
    col_index = np.hstack((edges[1], edges[0]))
    graph = sp.coo_matrix(
        (
            np.hstack((weights, weights, diagonal)),
            (np.hstack((row_index, diagonal_index)), np.hstack((col_index, diagonal_index))),
        ),
        (n_voxels, n_voxels),
        dtype=graph_dtype,
    )
    return _coo_or_dense(graph, return_as)

def _patch_extractor_result_valid(
    result: NDArray[np.float64],
    X: NDArray[np.float64],
    patch_size: PatchSize | None,
    max_patches: int | float | None,
) -> bool:
    values = np.asarray(X)
    patches = np.asarray(result)
    image_height, image_width = values.shape[1:3]
    if patch_size is None:
        patch_height, patch_width = image_height // 10, image_width // 10
    else:
        patch_height, patch_width = patch_size
    patches_per_image = _compute_n_patches(image_height, image_width, patch_height, patch_width, max_patches)
    expected_prefix = (values.shape[0] * patches_per_image, patch_height, patch_width)
    if values.ndim == 4 and values.shape[3] > 1:
        return bool(patches.shape == expected_prefix + (values.shape[3],) and np.all(np.isfinite(patches)))
    return bool(patches.shape == expected_prefix and np.all(np.isfinite(patches)))

@register_atom(witness_extract_patches_2d)
@icontract.require(lambda image: image.ndim in {2, 3}, "image must be 2D or 3D")
@icontract.require(lambda patch_size: len(patch_size) == 2, "patch_size must have two entries")
@icontract.require(lambda patch_size: patch_size[0] > 0 and patch_size[1] > 0, "patch dimensions must be positive")
@icontract.require(lambda image, patch_size: patch_size[0] <= image.shape[0], "patch height must fit image")
@icontract.require(lambda image, patch_size: patch_size[1] <= image.shape[1], "patch width must fit image")
@icontract.require(lambda max_patches: _valid_max_patches(max_patches), "max_patches must be None, a positive integer, or a fraction in (0, 1)")
@icontract.ensure(lambda result: result.ndim in {3, 4}, "patch tensor must be 3D or 4D")
@icontract.ensure(lambda result, patch_size: result.shape[1] == patch_size[0] and result.shape[2] == patch_size[1], "patch dimensions must match patch_size")
def extract_patches_2d(
    image: NDArray[np.float64],
    patch_size: PatchSize,
    *,
    max_patches: int | float | None = None,
    random_state: RandomStateArg = None,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array, check_random_state
    """Return all sliding 2D image patches, or a random subset of them."""
    image_height, image_width = image.shape[:2]
    patch_height, patch_width = patch_size

    checked_image = check_array(image, allow_nd=True)
    reshaped = checked_image.reshape((image_height, image_width, -1))
    n_channels = reshaped.shape[-1]
    extracted_patches = _extract_patches(
        reshaped,
        patch_shape=(patch_height, patch_width, n_channels),
        extraction_step=1,
    )
    n_patches = _compute_n_patches(
        image_height,
        image_width,
        patch_height,
        patch_width,
        max_patches,
    )
    if max_patches:
        rng = check_random_state(random_state)
        row_indices = rng.randint(image_height - patch_height + 1, size=n_patches)
        col_indices = rng.randint(image_width - patch_width + 1, size=n_patches)
        patches = extracted_patches[row_indices, col_indices, 0]
    else:
        patches = extracted_patches

    patch_array = patches.reshape(-1, patch_height, patch_width, n_channels)
    if patch_array.shape[-1] == 1:
        return np.asarray(patch_array.reshape((n_patches, patch_height, patch_width)))
    return np.asarray(patch_array)

@register_atom(witness_reconstruct_from_patches_2d)
@icontract.require(lambda patches: patches.ndim in {3, 4}, "patches must be 3D or 4D")
@icontract.require(lambda image_size: len(image_size) in {2, 3}, "image_size must have two or three entries")
@icontract.require(lambda image_size: image_size[0] > 0 and image_size[1] > 0, "image dimensions must be positive")
@icontract.require(lambda patches, image_size: patches.shape[1] <= image_size[0], "patch height must fit output image")
@icontract.require(lambda patches, image_size: patches.shape[2] <= image_size[1], "patch width must fit output image")
@icontract.ensure(lambda result, image_size: result.shape == tuple(image_size), "image shape must equal image_size")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "reconstructed pixels must be finite")
def reconstruct_from_patches_2d(
    patches: NDArray[np.float64],
    image_size: ImageSize,
) -> NDArray[np.float64]:
    """Average overlapping patches back into an image with the requested shape."""
    image_height, image_width = image_size[:2]
    patch_height, patch_width = patches.shape[1:3]
    image = np.zeros(image_size)
    n_h = image_height - patch_height + 1
    n_w = image_width - patch_width + 1
    for patch, (row, col) in zip(patches, product(range(n_h), range(n_w)), strict=False):
        image[row : row + patch_height, col : col + patch_width] += patch

    for row in range(image_height):
        for col in range(image_width):
            image[row, col] /= float(
                min(row + 1, patch_height, image_height - row)
                * min(col + 1, patch_width, image_width - col)
            )
    return np.asarray(image)

@register_atom(witness_img_to_graph)
@icontract.require(lambda img: img.ndim in {2, 3}, "img must be a 2D or 3D image")
@icontract.require(lambda img: img.size > 0, "img must contain at least one pixel")
@icontract.require(
    lambda mask, img: mask is None
    or mask.shape == img.shape
    or mask.shape == np.atleast_3d(img).shape,
    "mask must match the image shape",
)
@icontract.ensure(lambda result: result.shape[0] == result.shape[1], "graph must be square")
@icontract.ensure(lambda result: result.shape[0] > 0, "graph must contain at least one node")
def img_to_graph(
    img: NDArray[np.float64],
    *,
    mask: NDArray[np.bool_] | None = None,
    return_as: SparseReturn = sp.coo_matrix,
    dtype: npt.DTypeLike | None = None,
) -> GraphResult:
    """Build a pixel-neighbor graph whose edge weights are image differences."""
    image = np.atleast_3d(img)
    n_x, n_y, n_z = image.shape
    return _to_graph(
        int(n_x),
        int(n_y),
        int(n_z),
        mask=mask,
        image=image,
        return_as=return_as,
        dtype=dtype,
    )

@register_atom(witness_grid_to_graph)
@icontract.require(lambda n_x: n_x >= 1, "n_x must be positive")
@icontract.require(lambda n_y: n_y >= 1, "n_y must be positive")
@icontract.require(lambda n_z: n_z >= 1, "n_z must be positive")
@icontract.require(lambda mask, n_x, n_y, n_z: mask is None or mask.size == n_x * n_y * n_z, "mask must match grid size")
@icontract.ensure(lambda result: result.shape[0] == result.shape[1], "graph must be square")
@icontract.ensure(lambda result: result.shape[0] >= 0, "graph node count must be non-negative")
def grid_to_graph(
    n_x: int,
    n_y: int,
    n_z: int = 1,
    *,
    mask: NDArray[np.bool_] | None = None,
    return_as: SparseReturn = sp.coo_matrix,
    dtype: npt.DTypeLike = int,
) -> GraphResult:
    """Build a sparse adjacency graph for a 2D or 3D rectangular grid."""
    return _to_graph(
        int(n_x),
        int(n_y),
        int(n_z),
        mask=mask,
        return_as=return_as,
        dtype=dtype,
    )

@register_atom(witness_patch_extractor_transform)
@icontract.require(lambda X: X.ndim in {3, 4}, "X must contain grayscale or multichannel images")
@icontract.require(lambda X: X.shape[0] >= 1, "X must contain at least one image")
@icontract.require(lambda X: X.shape[1] >= 1 and X.shape[2] >= 1, "image dimensions must be positive")
@icontract.require(lambda X, patch_size: _valid_patch_size_for_images(X, patch_size), "patch_size must fit each image")
@icontract.require(lambda max_patches: _valid_max_patches(max_patches), "max_patches must be None, a positive integer, or a fraction in (0, 1)")
@icontract.ensure(lambda result, X, patch_size, max_patches: _patch_extractor_result_valid(result, X, patch_size, max_patches), "patch batch shape must match PatchExtractor semantics")
def patch_extractor_transform(
    X: NDArray[np.float64],
    *,
    patch_size: PatchSize | None = None,
    max_patches: int | float | None = None,
    random_state: RandomStateArg = None,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array, check_random_state
    """Extract patch tensors from a batch of grayscale or multichannel images."""
    checked_x = check_array(
        X,
        ensure_2d=False,
        allow_nd=True,
        ensure_min_samples=1,
        ensure_min_features=1,
    )
    rng = check_random_state(random_state)
    n_images, image_height, image_width = checked_x.shape[:3]
    if patch_size is None:
        resolved_patch_size = (image_height // 10, image_width // 10)
    else:
        resolved_patch_size = patch_size

    reshaped = np.reshape(checked_x, (n_images, image_height, image_width, -1))
    n_channels = reshaped.shape[-1]
    patch_height, patch_width = resolved_patch_size
    n_patches = _compute_n_patches(image_height, image_width, patch_height, patch_width, max_patches)
    patches_shape = (n_images * n_patches, patch_height, patch_width)
    if n_channels > 1:
        patches_shape += (n_channels,)

    patches = np.empty(patches_shape)
    for image_index, image in enumerate(reshaped):
        start = image_index * n_patches
        stop = (image_index + 1) * n_patches
        patches[start:stop] = extract_patches_2d(
            image,
            resolved_patch_size,
            max_patches=max_patches,
            random_state=rng,
        )
    return np.asarray(patches)
