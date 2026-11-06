"""Recipe boundary over caller-owned local shard stores and explicit numerics."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete local feature stores and numeric callables required')


@register_atom(concrete_only)
def execute(training: object, calibration: object, query: object, controls: dict,
            pool_function: object, gate_function: object, mean_function: object,
            threshold_fit_function: object, threshold_apply_function: object) -> dict:
    """Train full configured populations, fit held-out thresholds and predict.

    Caller supplies FeatureShards stores constructed from private runtime
    configuration. No paths or records belong in the catalog graph. Validates
    disjoint groups and uses bounded frame reads; labels, sparse training CSR,
    fitted state and scores remain resident. Numeric callables are explicit;
    model construction, optimization and population policy belong to this recipe.
    """
    from sciona.video_multilabel_sharded_training import execute as run
    return run(training, calibration, query, controls, pool_function=pool_function,
        gate_function=gate_function, mean_function=mean_function, threshold_fit_function=threshold_fit_function,
        threshold_apply_function=threshold_apply_function)
