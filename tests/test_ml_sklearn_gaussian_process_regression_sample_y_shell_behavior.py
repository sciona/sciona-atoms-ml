from __future__ import annotations

import numpy as np


def test_gp_regression_sample_y_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sample_y_shell import (
        gp_sample_y_result,
        gp_sample_y_use_multioutput_branch,
    )

    assert callable(gp_sample_y_use_multioutput_branch)
    assert callable(gp_sample_y_result)


def test_gp_sample_y_use_multioutput_branch_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sample_y_shell import (
        gp_sample_y_use_multioutput_branch,
    )

    assert gp_sample_y_use_multioutput_branch(np.array([0.5, -1.0], dtype=np.float64)) is False
    assert gp_sample_y_use_multioutput_branch(np.array([[0.5, -1.0], [1.25, 0.0]], dtype=np.float64)) is True


def test_gp_sample_y_result_matches_single_output_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sample_y_shell import (
        gp_sample_y_result,
    )
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sampling import (
        gp_sample_y_single_output,
    )

    y_mean = np.array([0.5, -1.0], dtype=np.float64)
    y_cov = np.array([[1.0, 0.2], [0.2, 1.5]], dtype=np.float64)

    result = gp_sample_y_result(y_mean, y_cov, n_samples=3, random_state=17)
    expected = gp_sample_y_single_output(y_mean, y_cov, n_samples=3, random_state=17)

    assert np.allclose(result, expected)
    assert result.shape == (2, 3)


def test_gp_sample_y_result_matches_multi_output_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sample_y_shell import (
        gp_sample_y_result,
    )
    from sciona.atoms.ml.sklearn.gaussian_process.regression_sampling import (
        gp_sample_y_multi_output,
    )

    y_mean = np.array([[0.5, -1.0], [1.25, 0.0]], dtype=np.float64)
    y_cov = np.stack(
        [
            np.array([[1.0, 0.2], [0.2, 1.5]], dtype=np.float64),
            np.array([[0.75, 0.1], [0.1, 0.5]], dtype=np.float64),
        ],
        axis=-1,
    )

    result = gp_sample_y_result(y_mean, y_cov, n_samples=4, random_state=3)
    expected = gp_sample_y_multi_output(y_mean, y_cov, n_samples=4, random_state=3)

    assert np.allclose(result, expected)
    assert result.shape == (2, 2, 4)
