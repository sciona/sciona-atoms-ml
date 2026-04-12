from __future__ import annotations

from pathlib import Path
import importlib

import numpy as np

from sciona.heuristic_metadata import resolve_external_atom_heuristic_metadata


def test_ml_sklearn_atom_namespace_import_smoke() -> None:
    module = importlib.import_module("sciona.atoms.ml.sklearn.images")
    assert hasattr(module, "extract_patches_2d")
    assert hasattr(module, "img_to_graph")


def test_ml_sklearn_patch_round_trip_fallback() -> None:
    module = importlib.import_module("sciona.atoms.ml.sklearn.images")
    image = np.arange(9, dtype=np.float64).reshape(3, 3)

    patches = module.extract_patches_2d(image, (2, 2))
    reconstructed = module.reconstruct_from_patches_2d(patches, (3, 3))

    assert patches.shape == (4, 2, 2)
    assert np.allclose(reconstructed, image)


def test_ml_sklearn_graph_wrappers_return_sparse_matrices() -> None:
    module = importlib.import_module("sciona.atoms.ml.sklearn.images")

    img_graph = module.img_to_graph(np.zeros((2, 2, 2), dtype=np.float64))
    grid_graph = module.grid_to_graph(2, 2, 1)

    assert img_graph.shape == (8, 8)
    assert grid_graph.shape == (4, 4)
    assert img_graph.nnz > 0
    assert grid_graph.nnz > 0


def test_ml_sklearn_probe_witnesses_return_structural_surrogates() -> None:
    module = importlib.import_module("sciona.probes.ml.sklearn.images")
    image = np.arange(9, dtype=np.float64).reshape(3, 3)
    patches = np.ones((4, 2, 2), dtype=np.float64)

    patch_surrogate = module.witness_image_patch_sampling_and_assembly(image, (2, 2))
    image_surrogate = module.witness_patches_to_image_reconstruction(patches, (3, 3))
    graph_surrogate = module.witness_3d_image_graph_materialization(
        np.zeros((2, 2, 2), dtype=np.float64)
    )
    grid_surrogate = module.witness_voxel_grid_graph_assembly(2, 2, 1)

    assert patch_surrogate.shape == (1, 2, 2)
    assert image_surrogate.shape == (3, 3)
    assert graph_surrogate.shape == (8, 8)
    assert grid_surrogate.shape == (4, 4)


def test_ml_sklearn_heuristic_metadata_resolves_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    metadata = resolve_external_atom_heuristic_metadata(
        "sciona.atoms.ml.sklearn.images.extract_patches_2d",
        provider_roots=(repo_root,),
    )

    assert metadata is not None
    assert metadata.atom_fqdn == "sciona.atoms.ml.sklearn.images.extract_patches_2d"
    assert metadata.heuristic_outputs[0].heuristic.heuristic_id == "boundary_discontinuity"
