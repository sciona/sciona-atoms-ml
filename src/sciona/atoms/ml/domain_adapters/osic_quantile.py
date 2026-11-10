"""Concrete graph interfaces for the source-specific OSIC quantile recipe.

These adapters do not confer catalog approval. Caller-owned mapped tables,
transductive population scope, feature coordinates and model state stay explicit.
The fixed model/schedule and baseline policy are distinct from the generic
distribution loss providers used by training. Imports are lazy so discovery
does not initialize TensorFlow or allocate models.
"""
from sciona.ghost.registry import register_atom


def witness_osic_quantile(*args, **kwargs):
    raise ValueError('Concrete mapped tables, numeric matrices and explicit state required')


@register_atom(witness_osic_quantile)
def assemble_quantile_populations(training, baseline, queries):
    """Assemble source populations after removing all duplicate training keys.

    Caller maps the exact neutral table fields and establishes availability.
    Each queried entity requires one supplied baseline. Encounter order matters.
    This performs joins and population tagging only, not feature fitting.
    """
    from sciona.osic_quantile_features import assemble_populations
    return assemble_populations(training, baseline, queries)


@register_atom(witness_osic_quantile)
def fit_joint_quantile_features(population, allow_transductive):
    """Fit the source feature recipe over the explicit joint population.

    Requires allow_transductive=True. Returns numeric matrices by role and
    ordered fitted feature metadata. A changed query population can change
    training features; this is not a train-only reusable scaler.
    """
    from sciona.osic_quantile_features import joint_population_features
    return joint_population_features(population, allow_transductive=allow_transductive)


@register_atom(witness_osic_quantile)
def train_prepared_quantile_folds(features, targets, query_features, columns,
                                  feature_coordinate, response_coordinate, seeds):
    """Train two fixed 600-epoch folds from aligned finite float32 matrices.

    Features (N,D), targets (N,1), queries (M,D); N>=2, M>=1, D>=1. Explicit
    ordered columns, coordinates and a tuple of two 31-bit seeds are required.
    Returns raw fold predictions, out-of-fold values, private prediction states
    and training records. No feature fitting or known-baseline overrides.
    """
    from sciona.osic_quantile_training import train_prepared_quantile_folds as train
    return train(features, targets, query_features, columns,
                 feature_coordinate=feature_coordinate, response_coordinate=response_coordinate, seeds=seeds)


@register_atom(witness_osic_quantile)
def replay_prepared_quantile_folds(states, features, columns, feature_coordinate, response_coordinate):
    """Restore both numeric fold states and return raw (2,M,3) predictions.

    Requires finite float32 features with the saved feature order and matching
    coordinates. No fitting, reduction or baseline override. Caller verifies
    state provenance; serialized state supports inference, not training resume.
    """
    from sciona.osic_quantile_training import replay_prepared_quantile_folds as replay
    return replay(states, features, columns, feature_coordinate, response_coordinate)


@register_atom(witness_osic_quantile)
def reduce_quantile_predictions(folds, observed_mask, observed_values):
    """Apply source fold averaging, midpoint/spread extraction and known values.

    Fold predictions (F,M,3) are finite float32/float64. Divide in input dtype
    before accumulating in float64. Preserve crossed/negative spreads. Caller
    supplies a boolean availability mask and aligned finite observed values;
    selected rows receive those values and uncertainty 0.1 before branch blend.
    This domain policy does not establish interval calibration.
    """
    from sciona.osic_predictions import quantile_predictions
    return quantile_predictions(folds, observed_mask, observed_values)
