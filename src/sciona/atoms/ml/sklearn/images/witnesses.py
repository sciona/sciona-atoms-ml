"""Ghost witnesses for sklearn image atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_extract_patches_2d(
    image: AbstractArray,
    patch_size: tuple[int, int],
    *,
    max_patches: int | float | None = None,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe the patch tensor shape produced by sliding window extraction."""
    del random_state
    if len(image.shape) not in {2, 3}:
        raise ValueError("image must be 2D or 3D")
    patch_height, patch_width = patch_size
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("patch dimensions must be positive")
    if patch_height > image.shape[0] or patch_width > image.shape[1]:
        raise ValueError("patch_size must fit inside image")
    total = (image.shape[0] - patch_height + 1) * (image.shape[1] - patch_width + 1)
    if max_patches is None:
        n_patches = total
    elif isinstance(max_patches, int):
        n_patches = min(max_patches, total)
    else:
        if not 0.0 < max_patches < 1.0:
            raise ValueError("fractional max_patches must be in (0, 1)")
        n_patches = int(max_patches * total)
    if len(image.shape) == 3:
        return AbstractArray(shape=(n_patches, patch_height, patch_width, image.shape[2]), dtype=image.dtype)
    return AbstractArray(shape=(n_patches, patch_height, patch_width), dtype=image.dtype)


def witness_reconstruct_from_patches_2d(
    patches: AbstractArray,
    image_size: tuple[int, ...],
) -> AbstractArray:
    """Describe the reconstructed image shape from a complete patch sequence."""
    if len(patches.shape) not in {3, 4}:
        raise ValueError("patches must be 3D or 4D")
    if len(image_size) not in {2, 3}:
        raise ValueError("image_size must have two or three entries")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image dimensions must be positive")
    if patches.shape[1] > image_size[0] or patches.shape[2] > image_size[1]:
        raise ValueError("patches must fit inside image_size")
    return AbstractArray(shape=tuple(image_size), dtype=patches.dtype)


def witness_patch_extractor_transform(
    X: AbstractArray,
    *,
    patch_size: tuple[int, int] | None = None,
    max_patches: int | float | None = None,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe batch image patch extraction from a stateless transformer."""
    del random_state
    if len(X.shape) not in {3, 4}:
        raise ValueError("X must contain grayscale or multichannel images")
    n_images, image_height, image_width = X.shape[:3]
    if patch_size is None:
        patch_height, patch_width = image_height // 10, image_width // 10
    else:
        patch_height, patch_width = patch_size
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("patch dimensions must be positive")
    if patch_height > image_height or patch_width > image_width:
        raise ValueError("patch_size must fit inside each image")
    total = (image_height - patch_height + 1) * (image_width - patch_width + 1)
    if max_patches is None:
        patches_per_image = total
    elif isinstance(max_patches, int):
        patches_per_image = min(max_patches, total)
    else:
        if not 0.0 < max_patches < 1.0:
            raise ValueError("fractional max_patches must be in (0, 1)")
        patches_per_image = int(max_patches * total)
    if len(X.shape) == 4 and X.shape[3] > 1:
        return AbstractArray(
            shape=(n_images * patches_per_image, patch_height, patch_width, X.shape[3]),
            dtype=X.dtype,
        )
    return AbstractArray(shape=(n_images * patches_per_image, patch_height, patch_width), dtype=X.dtype)


def witness_img_to_graph(
    img: AbstractArray,
    *,
    mask: AbstractArray | None = None,
    return_as: type | None = None,
    dtype: object | None = None,
) -> AbstractArray:
    """Describe the square image-neighbor graph produced from pixels."""
    del return_as, dtype
    if len(img.shape) not in {2, 3}:
        raise ValueError("img must be 2D or 3D")
    shape_3d = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)
    if mask is not None and mask.shape != img.shape and mask.shape != shape_3d:
        raise ValueError("mask must match the image shape")
    node_count = int(shape_3d[0] * shape_3d[1] * shape_3d[2])
    return AbstractArray(shape=(node_count, node_count), dtype=img.dtype)


def witness_grid_to_graph(
    n_x: int,
    n_y: int,
    n_z: int = 1,
    *,
    mask: AbstractArray | None = None,
    return_as: type | None = None,
    dtype: object | None = None,
) -> AbstractArray:
    """Describe the square graph produced from regular grid dimensions."""
    del return_as, dtype
    if n_x < 1 or n_y < 1 or n_z < 1:
        raise ValueError("grid dimensions must be positive")
    if mask is not None:
        mask_size = 1
        for dimension in mask.shape:
            mask_size *= dimension
        if mask_size != n_x * n_y * n_z:
            raise ValueError("mask must match grid size")
    node_count = int(n_x * n_y * n_z)
    return AbstractArray(shape=(node_count, node_count), dtype="float64")
