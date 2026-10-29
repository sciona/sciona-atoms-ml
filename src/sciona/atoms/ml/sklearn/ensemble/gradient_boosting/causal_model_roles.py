"""Recipe-only classifier role validation, with no prediction or fitting."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted
from sciona.ghost.registry import register_atom


def witness_validate_causal_model_roles(one_step_model, independence_model,
    direction_model, left_model, right_model) -> tuple:
    raise ValueError('Concrete fitted classifiers required to validate causal class roles')


@register_atom(witness_validate_causal_model_roles)
def validate_causal_model_roles(one_step_model: object, independence_model: object,
    direction_model: object, left_model: object, right_model: object) -> tuple:
    """Check source role label sets and return the same five models unchanged.

    Requires fitted sklearn GradientBoostingClassifiers. One-step labels are
    {-1,0,1}; direction labels are {-1,1}; independence/left/right are {0,1}.
    No probabilities are computed and no model is trained, copied or mutated.
    Actual generic prediction nodes check feature widths and finite outputs.
    Shape-only model metadata cannot prove these label contracts.
    """
    models=(one_step_model,independence_model,direction_model,left_model,right_model)
    for model,required in zip(models,({-1,0,1},{0,1},{-1,1},{0,1},{0,1})):
        if not isinstance(model,GradientBoostingClassifier):
            raise TypeError('GradientBoostingClassifier required')
        check_is_fitted(model)
        if model.classes_.dtype.kind not in 'iu' or set(model.classes_.tolist())!=required:
            raise ValueError('Classifier labels do not match the declared causal role')
    return models
