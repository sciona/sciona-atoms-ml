from __future__ import annotations

import numpy as np
import pytest


def test_dictionary_learning_minibatch_postfit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_postfit_shell import (
        dictionary_learning_minibatch_fit_return_self,
        dictionary_learning_minibatch_postfit_components,
        dictionary_learning_minibatch_postfit_n_iter,
        dictionary_learning_minibatch_postfit_n_steps,
    )

    assert callable(dictionary_learning_minibatch_postfit_components)
    assert callable(dictionary_learning_minibatch_postfit_n_steps)
    assert callable(dictionary_learning_minibatch_postfit_n_iter)
    assert callable(dictionary_learning_minibatch_fit_return_self)


def test_dictionary_learning_minibatch_postfit_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_postfit_shell import (
        dictionary_learning_minibatch_fit_return_self,
        dictionary_learning_minibatch_postfit_components,
        dictionary_learning_minibatch_postfit_n_iter,
        dictionary_learning_minibatch_postfit_n_steps,
    )

    dictionary = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(dictionary_learning_minibatch_postfit_components(dictionary), dictionary)
    assert dictionary_learning_minibatch_postfit_n_steps(17) == 17
    assert dictionary_learning_minibatch_postfit_n_iter(6.0) == pytest.approx(6.0)
    assert dictionary_learning_minibatch_fit_return_self("MiniBatchDictionaryLearning") == "MiniBatchDictionaryLearning"


def test_dictionary_learning_minibatch_postfit_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_minibatch_postfit_shell import (
        dictionary_learning_minibatch_fit_return_self,
        dictionary_learning_minibatch_postfit_components,
        dictionary_learning_minibatch_postfit_n_iter,
        dictionary_learning_minibatch_postfit_n_steps,
    )

    with pytest.raises(Exception):
        dictionary_learning_minibatch_postfit_components(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        dictionary_learning_minibatch_postfit_n_steps(0)

    with pytest.raises(Exception):
        dictionary_learning_minibatch_postfit_n_iter(0.0)

    with pytest.raises(Exception):
        dictionary_learning_minibatch_fit_return_self("")
