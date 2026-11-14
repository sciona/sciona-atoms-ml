"""Expose source preparation fields as explicit training graph ports."""
from sciona.ghost.registry import register_atom


def witness_training_projection(*args, **kwargs):
    raise ValueError('A concrete prepared image population is required')


@register_atom(witness_training_projection)
def unpack_image_training_population(prepared):
    """Borrow image groups, metadata and targets from source preparation.

    Require a plain dict with exactly entities, images, metadata, slopes, fits,
    time_coordinate and response_coordinate. Return a tuple of the images,
    metadata and slopes values without copying, conversion, fitting or sampling.
    Preserve array identity, dtype, order and aliasing. The upstream preparation
    and downstream trainer validate numeric alignment and shapes; this projection
    only validates field inventory. Diagnostics, entity order and coordinate
    declarations remain available in the original private mapping. Source field
    names are adapter policy; this does not infer a schema or establish units.
    """
    fields = {'entities', 'images', 'metadata', 'slopes', 'fits',
              'time_coordinate', 'response_coordinate'}
    if type(prepared) is not dict or set(prepared) != fields:
        raise ValueError('Exact prepared image population mapping required')
    return prepared['images'], prepared['metadata'], prepared['slopes']
