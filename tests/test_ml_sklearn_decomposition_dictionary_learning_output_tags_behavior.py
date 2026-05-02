from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning


def _components() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_dictionary_learning_output_tags_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_output_tags import (
        dictionary_learning_fit_return_self,
        dictionary_learning_minibatch_n_features_out,
        dictionary_learning_minibatch_preserves_dtype_tags,
        dictionary_learning_n_features_out,
        dictionary_learning_preserves_dtype_tags,
    )

    assert callable(dictionary_learning_fit_return_self)
    assert callable(dictionary_learning_n_features_out)
    assert callable(dictionary_learning_preserves_dtype_tags)
    assert callable(dictionary_learning_minibatch_n_features_out)
    assert callable(dictionary_learning_minibatch_preserves_dtype_tags)


def test_dictionary_learning_output_tags_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_output_tags import (
        dictionary_learning_fit_return_self,
        dictionary_learning_minibatch_n_features_out,
        dictionary_learning_minibatch_preserves_dtype_tags,
        dictionary_learning_n_features_out,
        dictionary_learning_preserves_dtype_tags,
    )

    components = _components()
    dictionary_model = DictionaryLearning(n_components=3, random_state=0)
    dictionary_model.components_ = components
    minibatch_model = MiniBatchDictionaryLearning(n_components=3, random_state=0)
    minibatch_model.components_ = components

    assert dictionary_learning_fit_return_self("DictionaryLearning") == "DictionaryLearning"
    assert dictionary_learning_n_features_out(components) == dictionary_model._n_features_out
    assert dictionary_learning_preserves_dtype_tags(("float64",)) == ("float64", "float32")
    assert dictionary_learning_minibatch_n_features_out(components) == minibatch_model._n_features_out
    assert dictionary_learning_minibatch_preserves_dtype_tags(("float32",)) == ("float64", "float32")


def test_dictionary_learning_output_tags_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_output_tags import (
        dictionary_learning_fit_return_self,
        dictionary_learning_minibatch_n_features_out,
        dictionary_learning_n_features_out,
        dictionary_learning_preserves_dtype_tags,
    )

    with pytest.raises(Exception):
        dictionary_learning_fit_return_self("")

    with pytest.raises(Exception):
        dictionary_learning_n_features_out(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        dictionary_learning_minibatch_n_features_out(np.array([], dtype=np.float64))

    with pytest.raises(Exception):
        dictionary_learning_preserves_dtype_tags(())
