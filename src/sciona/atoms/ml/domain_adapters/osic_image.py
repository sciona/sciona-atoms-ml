"""Concrete source-specific image preparation and prediction policies.

These lazy interfaces do not confer catalog approval. Caller-owned tables,
encoded images, exclusions and model outputs stay explicit. Generic affine
fitting and weighted summation remain independently reusable dependencies;
source intensity, metadata and uncertainty policies are not generalized here.
"""
from sciona.ghost.registry import register_atom


def witness_osic_image(*args, **kwargs):
    raise ValueError('Concrete mapped inputs and explicit numeric predictions required')


@register_atom(witness_osic_image)
def prepare_image_training_population(table, encoded_planes, excluded_entities,
                                      time_coordinate, response_coordinate):
    """Prepare source training images, metadata and longitudinal slope targets.

    Fit all mapped entities before applying caller-supplied exclusions. Preserve
    first-encounter entity order and first-row metadata. Decode eligible images
    in caller order, with the source training rescale/resize policy. Invalid
    eligible images raise; excluded payloads are not decoded. Return private
    image groups, metadata, slopes, fit diagnostics and coordinate declarations.
    No model construction, training or sampling occurs. Disable OpenCV workers
    before execution; the provisioned decoder supports only qualified formats.
    """
    from sciona.osic_image_population import prepare_training_population
    return prepare_training_population(table, encoded_planes,
        excluded_entities=excluded_entities, time_coordinate=time_coordinate,
        response_coordinate=response_coordinate)


@register_atom(witness_osic_image)
def prepare_image_slice_inputs(encoded_planes, numeric, category, status):
    """Prepare selected encoded slices for one entity without invoking a model.

    Caller supplies ordered unique (name, bytes) records and scalar metadata.
    Select source ordinal/count positions before decoding; retain at least two
    slices. Ignore image rescale metadata and apply the source inference
    normalization, resize and case-sensitive category policy. Return images
    (S,512,512,1), metadata (S,4) and selected int64 positions. Unselected payloads
    are not decoded. Disable OpenCV workers; caller owns order and availability.
    """
    from sciona.dicom_planes import decode_plane
    from sciona.osic_image_inference import prepare_slice_inputs
    return prepare_slice_inputs(encoded_planes, numeric, category, status,
        decode_selected=lambda payload: decode_plane(payload, read_rescale=False)['pixels'])


@register_atom(witness_osic_image)
def reduce_image_predictions(slice_slopes, baseline_time, baseline_value,
                             baseline_uncertainty, query_subject, query_time):
    """Reduce per-entity slice slopes to source point and uncertainty predictions.

    Require at least two finite scalar slice predictions per entity, aligned
    finite baseline vectors and integer entity positions for queries. Preserve
    input-dtype median rounding and source intercept-first arithmetic. Time and
    slope units must agree. The uncertainty heuristic can be negative and is
    not calibrated. No model execution, entity joining or availability inference.
    """
    from sciona.osic_predictions import image_trajectories
    return image_trajectories(slice_slopes, baseline_time, baseline_value,
        baseline_uncertainty, query_subject, query_time)


@register_atom(witness_osic_image)
def blend_keyed_predictions(image_keys, image_values, quantile_keys, quantile_values):
    """Align two prediction branches by keys and apply source weights 0.45/0.55.

    Both branches require identical nonempty unique string key sets and finite
    real (M,2) point/uncertainty matrices. Return sorted keys and a float64 matrix
    using the generic ordered weighted-sum provider. Caller establishes shared
    units and applicability; this source weighting is not a learned ensemble.
    """
    from sciona.osic_predictions import blend_predictions
    return blend_predictions(image_keys, image_values, quantile_keys, quantile_values)
