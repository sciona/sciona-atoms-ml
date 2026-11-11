"""Concrete model prediction with explicit native batching and loader limits.

Application feature construction, units, model architecture and model-state
authenticity remain caller responsibilities. Importing this module does not
initialize TensorFlow. This interface does not confer catalog approval.
"""
from sciona.ghost.registry import register_atom


def witness_predict_bounded(*args, **kwargs):
    raise ValueError('A concrete provisioned legacy-Keras model and aligned arrays are required')


@register_atom(witness_predict_bounded)
def predict_bounded(model, inputs, batch_size):
    """Predict a single aligned floating output with bounded Keras batching.

    Model must be a caller-provisioned tf_keras.Model. Inputs are a nonempty
    list/tuple of finite float32/float64 arrays sharing a nonzero leading
    dimension. Preserve native Keras conversion, row order and partial batches.
    Require an explicit positive Python integer batch_size, excluding bool.
    Return one finite floating ndarray with the same leading dimension.

    No fitting, model loading, domain feature transformation or unit conversion
    occurs. Arbitrary custom model code remains caller-trusted; this operation
    is not a sandbox or a model authenticity check. Only concrete provisioned
    execution is supported. Caller limits model-compute threads and reserves
    one additional private dataset worker; dataset autotuning is disabled.
    """
    from sciona._runtime.ml.keras_bounded_prediction import make_predictor
    return make_predictor(model, batch_size=batch_size)(inputs)
