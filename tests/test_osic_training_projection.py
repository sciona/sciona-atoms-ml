import numpy as np
import pytest

from sciona.atoms.ml.domain_adapters.osic_training_projection import unpack_image_training_population


def test_projection_preserves_aliases_and_leaves_diagnostics_untouched():
    images = [np.zeros((2, 2, 2, 1), dtype=np.float32)]
    metadata = np.arange(4, dtype=np.float64).reshape(1, 4)
    slopes = np.array([2.0], dtype=np.float64)
    diagnostics = object()
    prepared = dict(entities=['synthetic'], images=images, metadata=metadata,
                    slopes=slopes, fits=diagnostics, time_coordinate='synthetic-time',
                    response_coordinate='synthetic-response')
    before = prepared.copy()
    result = unpack_image_training_population(prepared)
    assert all(actual is expected for actual, expected in zip(result, (images, metadata, slopes)))
    assert all(prepared[key] is value for key, value in before.items())
    result[1][0, 0] = 7
    assert metadata[0, 0] == 7
    assert prepared['fits'] is diagnostics
    for key in prepared:
        incomplete = dict(prepared)
        del incomplete[key]
        with pytest.raises(ValueError):
            unpack_image_training_population(incomplete)
    with pytest.raises(ValueError):
        unpack_image_training_population(dict(prepared, unexpected=True))
    with pytest.raises(ValueError):
        unpack_image_training_population(None)
