"""Explicit numeric dtype and structured feature-label conversion boundaries."""
import json
import numpy as np
from sciona.ghost.registry import register_atom


def witness_typed_inputs(*args, **kwargs):
    raise ValueError('Concrete arrays or structured labels and explicit conversion required')


@register_atom(witness_typed_inputs)
def cast_finite_array(values, dtype):
    """Copy a finite float32/float64 ndarray to explicit float32 or float64 dtype.

    Require a nonempty plain ndarray and an exact dtype string. Preserve shape
    and element order and always allocate independent storage. Use NumPy native
    conversion; narrowing may round or underflow to zero. Reject nonfinite input
    or output, including narrowing overflow. No normalization, unit conversion,
    clipping or feature inference. Caller decides whether precision loss is
    acceptable and preserves array-axis, coordinate and availability meanings.
    """
    if (type(values) is not np.ndarray or values.size==0
            or values.dtype not in (np.dtype('float32'),np.dtype('float64'))
            or not np.isfinite(values).all()):
        raise ValueError('Nonempty finite plain float32/float64 ndarray required')
    if type(dtype) is not str or dtype not in ('float32','float64'):
        raise ValueError('Explicit float32 or float64 dtype string required')
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        try:result=values.astype(dtype,copy=True)
        except FloatingPointError as exc:raise ValueError('Conversion must remain finite') from exc
    if not np.isfinite(result).all():raise ValueError('Conversion must remain finite')
    return result


@register_atom(witness_typed_inputs)
def encode_feature_descriptors(descriptors):
    """Encode ordered structured feature labels as compact JSON strings.

    Require a nonempty plain list of nonempty plain lists of nonblank Python
    strings, with unique complete descriptors. Preserve descriptor/component
    order and exact spelling, including surrounding whitespace. Use standard
    JSON ASCII escaping and compact separators; do not join with a delimiter
    that could collide with label content. Return a new list of unique strings.
    Caller establishes column alignment and feature meanings; this does not
    infer schemas, sort labels, cast arrays or persist private metadata.
    """
    if type(descriptors) is not list or not descriptors:
        raise ValueError('Nonempty ordered descriptor list required')
    if any(type(item) is not list or not item or
           any(type(part) is not str or not part.strip() for part in item)
           for item in descriptors):
        raise ValueError('Nonempty lists of nonblank strings required')
    encoded=[json.dumps(item,separators=(',',':'),ensure_ascii=True) for item in descriptors]
    if len(set(encoded))!=len(encoded):raise ValueError('Unique feature descriptors required')
    return encoded
