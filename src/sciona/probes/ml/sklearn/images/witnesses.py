"""Probe-side structural witnesses for the sklearn ML surface."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


def witness_image_patch_sampling_and_assembly(
    image: np.ndarray,
    patch_size: tuple[int, int],
    max_patches: int | float | None = None,
    random_state: int | np.random.RandomState | np.random.Generator | None = None,
) -> np.ndarray:
    """Validate metadata for extracting 2D image patches."""
    del max_patches, random_state
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch_size must be positive")
    if patch_h > image.shape[0] or patch_w > image.shape[1]:
        raise ValueError("patch_size cannot exceed image dimensions")
    return np.zeros((1, patch_h, patch_w), dtype=image.dtype)


def witness_patches_to_image_reconstruction(
    patches: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Validate metadata for reconstructing an image from patches."""
    if patches.ndim != 3:
        raise ValueError("patches must be 3D")
    image_h, image_w = int(image_size[0]), int(image_size[1])
    if image_h <= 0 or image_w <= 0:
        raise ValueError("image_size must be positive")
    return np.zeros((image_h, image_w), dtype=patches.dtype)


def witness_3d_image_graph_materialization(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    return_as: type[sp.spmatrix] | None = None,
    dtype: np.dtype[Any] | None = None,
) -> sp.spmatrix:
    """Validate metadata for converting an image volume into a sparse graph."""
    del mask, return_as, dtype
    if img.ndim != 3:
        raise ValueError("img must be 3D")
    total = int(np.prod(img.shape))
    return sp.csr_matrix((total, total), dtype=img.dtype)


def witness_voxel_grid_graph_assembly(
    n_x: int,
    n_y: int,
    n_z: int = 1,
    mask: np.ndarray | None = None,
    return_as: type[sp.spmatrix] | None = None,
    dtype: np.dtype[Any] | None = None,
) -> sp.spmatrix:
    """Validate metadata for building a sparse graph over a voxel grid."""
    del mask, return_as, dtype
    if n_x <= 0 or n_y <= 0 or n_z <= 0:
        raise ValueError("grid dimensions must be positive")
    total = int(n_x * n_y * n_z)
    return sp.csr_matrix((total, total), dtype=np.float64)


_MODULE = "sciona.probes.ml.sklearn.images.witnesses"

IMAGES_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.witness_image_patch_sampling_and_assembly", _MODULE, "witness_image_patch_sampling_and_assembly"),
    ProbeTarget(f"{_MODULE}.witness_patches_to_image_reconstruction", _MODULE, "witness_patches_to_image_reconstruction"),
    ProbeTarget(f"{_MODULE}.witness_3d_image_graph_materialization", _MODULE, "witness_3d_image_graph_materialization"),
    ProbeTarget(f"{_MODULE}.witness_voxel_grid_graph_assembly", _MODULE, "witness_voxel_grid_graph_assembly"),
)


def probe_records() -> list[dict[str, object]]:
    """Return structural probe records for the sklearn image witness surface."""
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in IMAGES_PROBE_TARGETS
    ]
