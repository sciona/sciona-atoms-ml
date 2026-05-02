from __future__ import annotations

from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF


def test_gp_regression_kernel_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_kernel_shell import (
        gp_fit_kernel,
        gp_predict_prior_kernel,
        gp_regression_requires_fit_tag,
    )

    assert callable(gp_fit_kernel)
    assert callable(gp_predict_prior_kernel)
    assert callable(gp_regression_requires_fit_tag)


def test_gp_fit_kernel_defaults_or_clones() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_kernel_shell import (
        gp_fit_kernel,
    )

    default_kernel = gp_fit_kernel()
    expected = C(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    assert repr(default_kernel) == repr(expected)

    supplied = C(2.0) * RBF(3.0)
    resolved = gp_fit_kernel(supplied)
    assert repr(resolved) == repr(supplied)
    assert resolved is not supplied


def test_gp_predict_prior_kernel_defaults_or_reuses() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_kernel_shell import (
        gp_predict_prior_kernel,
    )

    default_kernel = gp_predict_prior_kernel()
    expected = C(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    assert repr(default_kernel) == repr(expected)

    supplied = C(2.0) * RBF(3.0)
    resolved = gp_predict_prior_kernel(supplied)
    assert resolved is supplied


def test_gp_regression_requires_fit_tag_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_kernel_shell import (
        gp_regression_requires_fit_tag,
    )

    assert gp_regression_requires_fit_tag(True) is False
    assert gp_regression_requires_fit_tag(False) is False
