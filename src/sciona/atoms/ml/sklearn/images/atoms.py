"""sklearn-backed atom wrappers with local fallbacks.

The upstream `sklearn.feature_extraction.image` module is used when available,
but this environment does not currently have a compatible wheel. The fallback
implementations keep the provider importable and preserve the clear surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import importlib

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

_SKLEARN_IMAGE_MODULE: Any | None = None
try:  # pragma: no cover - exercised only when sklearn imports cleanly
    _SKLEARN_IMAGE_MODULE = importlib.import_module("sklearn.feature_extraction.image")
except Exception:
    _SKLEARN_IMAGE_MODULE = None

PatchSize = tuple[int, int]
ImageSize = tuple[int, int]
OptionalDType = npt.DTypeLike | None
OptionalRandomState = int | np.random.RandomState | np.random.Generator | None
OptionalSparseCtor = type[sp.spmatrix] | None


def _coerce_patch_size(patch_size: Sequence[int]) -> PatchSize:
    if len(patch_size) != 2:
        raise ValueError("patch_size must contain exactly two dimensions")
    height = int(patch_size[0])
    width = int(patch_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("patch_size must be positive")
    return height, width


def _coerce_rng(
    random_state: OptionalRandomState,
) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        seed = int(random_state.randint(0, 2**32 - 1))
        return np.random.default_rng(seed)
    return np.random.default_rng(random_state)


def _extract_patches_2d_fallback(
    image: np.ndarray,
    patch_size: PatchSize,
    *,
    max_patches: int | float | None = None,
    random_state: OptionalRandomState = None,
) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    patch_h, patch_w = _coerce_patch_size(patch_size)
    if patch_h > image.shape[0] or patch_w > image.shape[1]:
        raise ValueError("patch_size cannot exceed image dimensions")

    patches: list[np.ndarray] = []
    for row in range(image.shape[0] - patch_h + 1):
        for col in range(image.shape[1] - patch_w + 1):
            patches.append(np.array(image[row : row + patch_h, col : col + patch_w], copy=True))

    if max_patches is not None:
        if isinstance(max_patches, float):
            if not 0.0 < max_patches <= 1.0:
                raise ValueError("max_patches as a fraction must be in (0, 1]")
            limit = max(1, int(round(len(patches) * max_patches)))
        else:
            limit = max(0, int(max_patches))
        if limit < len(patches):
            rng = _coerce_rng(random_state)
            indices = np.sort(rng.choice(len(patches), size=limit, replace=False))
            patches = [patches[index] for index in indices]

    if not patches:
        return np.empty((0, patch_h, patch_w), dtype=image.dtype)
    return np.stack(patches, axis=0)


def _reconstruct_from_patches_2d_fallback(
    patches: np.ndarray,
    image_size: ImageSize,
) -> np.ndarray:
    patches = np.asarray(patches)
    if patches.ndim != 3:
        raise ValueError("patches must be a 3D array")
    image_h, image_w = int(image_size[0]), int(image_size[1])
    if image_h <= 0 or image_w <= 0:
        raise ValueError("image_size must be positive")
    patch_h, patch_w = int(patches.shape[1]), int(patches.shape[2])
    grid_h = image_h - patch_h + 1
    grid_w = image_w - patch_w + 1
    expected = grid_h * grid_w
    if expected != patches.shape[0]:
        raise ValueError("patch count does not match the requested image size")

    reconstructed = np.zeros((image_h, image_w), dtype=patches.dtype)
    weights = np.zeros((image_h, image_w), dtype=np.int32)
    patch_index = 0
    for row in range(grid_h):
        for col in range(grid_w):
            reconstructed[row : row + patch_h, col : col + patch_w] += patches[patch_index]
            weights[row : row + patch_h, col : col + patch_w] += 1
            patch_index += 1
    return reconstructed / np.maximum(weights, 1)


def _adjacency_from_grid(
    grid_shape: tuple[int, ...],
    *,
    values: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    dtype: OptionalDType = None,
) -> sp.csr_matrix:
    total = int(np.prod(grid_shape))
    if total <= 0:
        raise ValueError("grid shape must be positive")

    value_array = None if values is None else np.asarray(values, dtype=np.float64)
    mask_array = None if mask is None else np.asarray(mask, dtype=bool)
    coordinates: list[tuple[int, int]] = []
    weights: list[float] = []

    def flat_index(index: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(index, grid_shape))

    for index in np.ndindex(grid_shape):
        if mask_array is not None and not bool(mask_array[index]):
            continue
        source = flat_index(index)
        for axis in range(len(grid_shape)):
            if index[axis] + 1 >= grid_shape[axis]:
                continue
            neighbor = list(index)
            neighbor[axis] += 1
            neighbor_t = tuple(neighbor)
            if mask_array is not None and not bool(mask_array[neighbor_t]):
                continue
            target = flat_index(neighbor_t)
            if value_array is None:
                weight = 1.0
            else:
                weight = 1.0 / (1.0 + abs(float(value_array[index]) - float(value_array[neighbor_t])))
            coordinates.extend(((source, target), (target, source)))
            weights.extend((weight, weight))

    if not coordinates:
        graph = sp.csr_matrix((total, total), dtype=dtype or np.float64)
    else:
        rows, cols = zip(*coordinates, strict=False)
        graph = sp.coo_matrix(
            (np.asarray(weights, dtype=dtype or np.float64), (rows, cols)),
            shape=(total, total),
        ).tocsr()
    if dtype is not None:
        graph = graph.astype(dtype)
    return graph


def extract_patches_2d(
    image: np.ndarray,
    patch_size: PatchSize,
    *,
    max_patches: int | float | None = None,
    random_state: OptionalRandomState = None,
) -> np.ndarray:
    """Extract fixed-size patches from a 2D image."""
    if _SKLEARN_IMAGE_MODULE is not None:
        result = _SKLEARN_IMAGE_MODULE.extract_patches_2d(
            image,
            patch_size,
            max_patches=max_patches,
            random_state=random_state,
        )
        return np.asarray(result)
    return _extract_patches_2d_fallback(
        image,
        patch_size,
        max_patches=max_patches,
        random_state=random_state,
    )


def reconstruct_from_patches_2d(patches: np.ndarray, image_size: ImageSize) -> np.ndarray:
    """Reconstruct an image by averaging overlapping patches."""
    if _SKLEARN_IMAGE_MODULE is not None:
        return np.asarray(_SKLEARN_IMAGE_MODULE.reconstruct_from_patches_2d(patches, image_size))
    return _reconstruct_from_patches_2d_fallback(patches, image_size)


def img_to_graph(
    img: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    return_as: OptionalSparseCtor = None,
    dtype: OptionalDType = None,
) -> sp.spmatrix:
    """Build a sparse adjacency graph from image pixels or voxels."""
    if _SKLEARN_IMAGE_MODULE is not None:
        graph = _SKLEARN_IMAGE_MODULE.img_to_graph(
            img,
            mask=mask,
            return_as=return_as or sp.coo_matrix,
            dtype=dtype,
        )
        return graph
    graph = _adjacency_from_grid(
        np.asarray(img).shape,
        values=np.asarray(img),
        mask=mask,
        dtype=dtype,
    )
    if return_as is not None:
        return return_as(graph)
    return graph


def grid_to_graph(
    n_x: int,
    n_y: int,
    n_z: int = 1,
    *,
    mask: np.ndarray | None = None,
    return_as: OptionalSparseCtor = None,
    dtype: OptionalDType = None,
) -> sp.spmatrix:
    """Build a sparse lattice graph for a regular 2D or 3D grid."""
    if _SKLEARN_IMAGE_MODULE is not None:
        graph = _SKLEARN_IMAGE_MODULE.grid_to_graph(
            n_x,
            n_y,
            n_z=n_z,
            mask=mask,
            return_as=return_as or sp.coo_matrix,
            dtype=dtype,
        )
        return graph
    graph = _adjacency_from_grid(
        (int(n_x), int(n_y), int(n_z)),
        mask=mask,
        dtype=dtype,
    )
    if return_as is not None:
        return return_as(graph)
    return graph
