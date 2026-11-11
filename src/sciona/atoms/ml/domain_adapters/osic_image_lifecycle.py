"""Concrete fixed-architecture image model and numeric inference-state boundaries.

The source B5 topology is explicit adapter policy. Generic bounded prediction
remains independently selectable. These lazy interfaces do not confer approval.
Models, weights, coordinates and state directories remain private runtime data.
"""
from sciona.ghost.registry import register_atom


def witness_image_lifecycle(*args, **kwargs):
    raise ValueError('Concrete provisioned model construction or private numeric state required')


@register_atom(witness_image_lifecycle)
def build_image_model():
    """Construct the fixed random-initialized B5 slope model without fitting.

    Require the qualified efficientnet 1.1.0 and legacy-Keras runtime. Inputs
    are image tensors (N,512,512,1) and metadata (N,4), output is (N,1).
    Preserve GaussianNoise(0.2), Dropout(0.5) and the source linear head.
    No pretrained weights, optimizer or normalization are supplied. Construction
    consumes caller RNG state; caller owns seeding, units and feature policy.
    """
    from sciona.osic_image_model import build_model
    return build_model()


@register_atom(witness_image_lifecycle)
def export_image_prediction_state(model, directory, image_coordinate,
                                  metadata_coordinate, response_coordinate):
    """Export fixed-model finite float32 weights and a strict numeric manifest.

    Require the exact local topology, a positive optimizer update counter and
    explicit nonempty coordinate declarations. The counter is not evidence of
    complete training. Write only into a new caller-owned private directory;
    return its manifest. Constructor-based validation consumes RNG state: run
    outside live training. No optimizer, callbacks or RNG state are exported,
    so the result supports inference only. Writes are not an atomic transaction;
    failed I/O may leave a partial directory. Caller controls paths and provenance.
    """
    from sciona.osic_image_state import export_prediction_state
    return export_prediction_state(model, directory,
        image_coordinate=image_coordinate, metadata_coordinate=metadata_coordinate,
        response_coordinate=response_coordinate)


@register_atom(witness_image_lifecycle)
def restore_image_prediction_state(directory, image_coordinate,
                                   metadata_coordinate, response_coordinate):
    """Restore the fixed local model from private numeric inference state.

    Validate exact manifest fields, caller coordinates, local model-code and
    graph identities, archive digest and ordered finite float32 weight shapes.
    Load NPZ with pickle disabled and reconstruct locally; do not deserialize
    executable model graphs or fit. Return an uncompiled model without optimizer
    state. Construction consumes RNG state. Caller establishes authenticity,
    trusted directory access and stable files; hashes alone are not provenance.
    """
    from sciona.osic_image_state import restore_prediction_state
    return restore_prediction_state(directory,
        image_coordinate=image_coordinate, metadata_coordinate=metadata_coordinate,
        response_coordinate=response_coordinate)
