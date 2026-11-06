"""Explicit PLAsTiCC policy stages; local registration is not approval."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete complete inputs required')


@register_atom(concrete_only)
def prepare(payload: dict, amplitude_function: object, fit_function: object, kernel_function: object, scale_function: object, noise_function: object, condition_function: object, predict_function: object) -> object:
    """Validate the invocation and bind numeric GP operations explicitly."""
    from sciona.decomposed_plasticc_observations import prepare as run
    return run(payload, dict(amplitude=amplitude_function, fit=fit_function, kernel=kernel_function,
        scale=scale_function, noise=noise_function, condition=condition_function, predict=predict_function))


@register_atom(concrete_only)
def augment(prepared: object) -> object:
    """Execute every configured augmentation attempt with retained retry policy."""
    from sciona.decomposed_plasticc_training import augment as run
    return run(prepared)


@register_atom(concrete_only)
def featurizer(prepared: object) -> object:
    """Construct the explicit astronomy feature policy after root validation."""
    from sciona.plasticc_features import PlasticcFeaturizer
    return PlasticcFeaturizer()


@register_atom(concrete_only)
def extract_features(population: object, featurizer: object) -> object:
    """Extract and select features for every supplied object."""
    from sciona.decomposed_plasticc_training import extract_features as run
    return run(population, featurizer)


@register_atom(concrete_only)
def assign_folds(population: object, prepared: object, fold_function: object) -> object:
    """Project original references into the supplied generic fold operation."""
    from sciona.decomposed_plasticc_training import assign_folds as run
    return run(population, prepared, fold_function)


@register_atom(concrete_only)
def training_weights(population: object, prepared: object, balance_function: object) -> tuple[object, object]:
    """Choose explicit flat or redshift policy and task class weights."""
    from sciona.decomposed_plasticc_training import training_weights as run
    return run(population, prepared, balance_function)


@register_atom(concrete_only)
def training_inputs(population: object, prepared: object) -> tuple[object, dict]:
    """Expose labels and all configured training controls with one worker."""
    parameters = dict(prepared.training_parameters)
    parameters.setdefault('n_jobs', 1)
    if type(parameters['n_jobs']) is not int or parameters['n_jobs'] != 1:
        raise ValueError('Serial training requires n_jobs=1')
    return population.metadata['class'], parameters


@register_atom(concrete_only)
def fit_fold(train_features: object, train_classes: object, train_weights: object, validation_features: object, validation_classes: object, validation_weights: object, parameters: dict) -> object:
    """Fit one weighted multiclass fold with explicit model controls."""
    from sciona.plasticc_training import fit_lightgbm_classifier
    if type(parameters.get('n_jobs')) is not int or parameters['n_jobs'] != 1:
        raise ValueError('Serial training requires n_jobs=1')
    return fit_lightgbm_classifier(train_features, train_classes, train_weights,
        validation_features, validation_classes, validation_weights, **parameters)


@register_atom(concrete_only)
def fit_folds(features: object, labels: object, weights: object, folds: object, parameters: dict, fit_function: object) -> object:
    """Execute every fold using the exact supplied one-fold provider."""
    from sciona.decomposed_plasticc_training import fit_folds as run
    def fit(a, b, c, d, e, f, **controls):
        return fit_function(a, b, c, d, e, f, controls)
    return run(features, labels, weights, folds, parameters, fit)


@register_atom(concrete_only)
def complete(prepared: object, population: object, featurizer: object, result: object, weights: object, class_weights: object) -> object:
    """Retain every model, OOF score, importance and source loss diagnostic."""
    from sciona.decomposed_plasticc_training import complete as run
    return run(prepared, population, featurizer, result, weights, class_weights)


@register_atom(concrete_only)
def prediction_inputs(completed: object) -> tuple[object, object, list, object]:
    """Expose query population, feature policy, all models and class order."""
    classifier = completed.classifier
    return completed.prediction, classifier.featurizer, classifier.classifiers, classifier.train_predictions.columns


@register_atom(concrete_only)
def task_policy(population: object, probabilities: object) -> object:
    """Apply galactic masks and the task-tuned unknown-class formula."""
    from sciona.decomposed_plasticc_prediction import task_policy as run
    return run(population, probabilities)


@register_atom(concrete_only)
def encode(final: object) -> dict:
    """Validate final probability mass and encode the runtime response."""
    from sciona.decomposed_plasticc_prediction import encode as run
    return run(final)
