"""Explicit loading boundary for caller-trusted legacy-Keras checkpoints."""
from sciona.ghost.registry import register_atom


def witness_load_trusted_keras_checkpoint(*args, **kwargs):
    raise ValueError('A concrete caller-trusted checkpoint and provisioned runtime are required')


@register_atom(witness_load_trusted_keras_checkpoint)
def load_trusted_keras_checkpoint(checkpoint, custom_objects):
    """Load one selected trusted model file with native legacy-Keras semantics.

    Require an existing caller-owned checkpoint file and an explicit dictionary
    of string names to caller-trusted custom objects (empty is allowed). Caller
    provisions tf_keras and any registered custom layers before invocation.
    Restore using compile=True, preserving native model, loss and optimizer
    loading behavior. Return the model without fitting, prediction, selection,
    normalization or architecture substitution. Native loader failures propagate.

    This operation deserializes executable model definitions; the caller must
    establish file authenticity, stable private storage and trusted custom code.
    It is not a sandbox, a numeric-only loader, or an untrusted-file validator.
    A checkpoint path does not prove best-epoch selection or completed training.
    Missing optimizer state may trigger native warnings/reinitialization; callers
    requiring it must verify the restored state. Callback, RNG and data-iterator
    state are not restored, so this does not promise exact training continuation.
    Bound native threads before use. Models and paths remain private runtime data.
    """
    from pathlib import Path

    if not isinstance(checkpoint, (str, Path)):
        raise ValueError('A concrete checkpoint file path is required')
    if type(custom_objects) is not dict or any(type(key) is not str or not key for key in custom_objects):
        raise ValueError('Explicit named custom objects dictionary required')
    path = Path(checkpoint)
    if not path.is_file():
        raise ValueError('Checkpoint must be an existing file')
    import tf_keras as keras
    return keras.models.load_model(path, custom_objects=custom_objects, compile=True)
