"""Explicit independent coordinate declarations for the two inference branches."""
from sciona.ghost.registry import register_atom


def witness_coordinates(*args, **kwargs):
    raise ValueError('Concrete caller coordinate declarations required')


@register_atom(witness_coordinates)
def route_response_coordinates(image_response_coordinate, quantile_response_coordinate):
    """Return image and quantile coordinate strings independently, in that order.

    Require nonblank exact Python strings and preserve spelling without trimming.
    The image model produces a slope; the quantile model produces response
    values. Each declaration must match its saved state. Caller establishes the
    relation between slope, time and response units before trajectory reduction
    and blending. No unit inference, conversion or compatibility proof occurs.
    Equal declarations are permitted; equality does not establish compatibility.
    """
    values=(image_response_coordinate, quantile_response_coordinate)
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError('Explicit nonblank image and quantile coordinates required')
    return values
