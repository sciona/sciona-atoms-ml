from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import DictionaryLearning
from sklearn.utils.validation import validate_data


def _data() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_dictionary_learning_fit_transform_prelude_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_fit_transform_prelude import (
        dictionary_learning_fit_transform_method,
        dictionary_learning_fit_transform_n_components,
        dictionary_learning_fit_transform_require_positive_coding_supported,
        dictionary_learning_fit_transform_validated_data,
    )

    assert callable(dictionary_learning_fit_transform_require_positive_coding_supported)
    assert callable(dictionary_learning_fit_transform_method)
    assert callable(dictionary_learning_fit_transform_validated_data)
    assert callable(dictionary_learning_fit_transform_n_components)


def test_dictionary_learning_fit_transform_prelude_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_fit_transform_prelude import (
        dictionary_learning_fit_transform_method,
        dictionary_learning_fit_transform_n_components,
        dictionary_learning_fit_transform_require_positive_coding_supported,
        dictionary_learning_fit_transform_validated_data,
    )

    X = _data()
    model = DictionaryLearning(n_components=None, fit_algorithm="cd", positive_code=True, random_state=0)
    validated = validate_data(model, X)

    assert dictionary_learning_fit_transform_require_positive_coding_supported("cd", True) is True
    assert dictionary_learning_fit_transform_method("cd") == "lasso_cd"
    assert np.array_equal(dictionary_learning_fit_transform_validated_data(X), validated)
    assert dictionary_learning_fit_transform_n_components(validated, None) == validated.shape[1]
    assert dictionary_learning_fit_transform_n_components(validated, 2) == 2


def test_dictionary_learning_fit_transform_prelude_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.dictionary_learning_fit_transform_prelude import (
        dictionary_learning_fit_transform_method,
        dictionary_learning_fit_transform_n_components,
        dictionary_learning_fit_transform_require_positive_coding_supported,
        dictionary_learning_fit_transform_validated_data,
    )

    with pytest.raises(Exception):
        dictionary_learning_fit_transform_require_positive_coding_supported("lars", True)

    with pytest.raises(Exception):
        dictionary_learning_fit_transform_method("omp")

    with pytest.raises(Exception):
        dictionary_learning_fit_transform_validated_data(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        dictionary_learning_fit_transform_n_components(_data(), 0)
