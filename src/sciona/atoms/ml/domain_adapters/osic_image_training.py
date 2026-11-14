"""Source image training over explicit prepared arrays and private checkpoints.

Raw decoding, longitudinal fitting, exclusions and coordinate interpretation
remain upstream adapters. Numeric inference-state export remains a separate
operation. Importing this provider neither trains nor grants catalog approval.
"""
from sciona.ghost.registry import register_atom


def witness_image_training(*args, **kwargs):
    raise ValueError('Concrete prepared image groups and qualified training runtime required')


@register_atom(witness_image_training)
def train_prepared_image_population(images, metadata, slopes, seed, checkpoint_directory):
    """Run the fixed source schedule and return final model, history, best path.

    Require ordered nonempty image groups (members,512,512,1) sharing float32
    or float64, finite floating metadata (groups,4), and slopes (groups,).
    The caller supplies matching units, source preparation and availability
    policy. Seed must be a Python int in the uint32 range; bool is rejected.

    Construct random B5, then run thirty epochs of thirty-two training and
    sixteen same-population validation batches, each of eight sampled members.
    Preserve legacy Adam, MAE and source callback/checkpoint policies. Sequential
    sampling does not reproduce historical prefetch or unused-backbone RNG draws.
    Reserve one compute core and bound native pools before invocation; this
    training loop does not create a dataset loader.

    Checkpoint directory must be new and caller-owned. It receives a private
    full-model HDF5 checkpoint, which must not be committed or published.
    Return the final in-memory model, thirty history records, and the selected
    best checkpoint path. The final model is not necessarily the selected best
    model. Downstream selection and numeric export must remain explicit.
    Training is not resumable here; failure may leave partial private files.
    Same-population validation does not establish generalization or transfer.
    """
    from sciona._runtime.ml.osic_image_training import train
    return train(images, metadata, slopes, seed=seed, checkpoint_directory=checkpoint_directory)
