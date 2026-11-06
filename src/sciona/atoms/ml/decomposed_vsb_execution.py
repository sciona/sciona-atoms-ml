"""Explicit VSB numeric providers and recipe orchestration boundaries."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args,**kwargs):
    raise ValueError("Concrete complete inputs required")


@register_atom(concrete_only)
def unpack(payload: dict) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict,dict]:
    """Explicit unpack operation for qualified runtime composition."""
    from sciona.decomposed_vsb_contract import unpack as run
    return run(payload)


@register_atom(concrete_only)
def fundamental(values: np.ndarray) -> np.ndarray:
    """Explicit fundamental operation for qualified runtime composition."""
    from sciona.vsb_measurement import fundamental as run
    return run(values)


@register_atom(concrete_only)
def baseline_arguments(values: np.ndarray,retention: float) -> tuple[np.ndarray,np.ndarray,np.ndarray,int,np.ndarray]:
    """Explicit baseline arguments operation for qualified runtime composition."""
    from sciona.decomposed_vsb_features import baseline_arguments as run
    return run(values,retention)


@register_atom(concrete_only)
def baseline_residual(filtered: np.ndarray) -> np.ndarray:
    """Explicit baseline residual operation for qualified runtime composition."""
    from sciona.decomposed_vsb_features import baseline_residual as run
    return run(filtered)


@register_atom(concrete_only)
def window_maxima(values: np.ndarray,window: int) -> np.ndarray:
    """Explicit window maxima operation for qualified runtime composition."""
    from sciona.vsb_peaks import window_maxima as run
    return run(values,window=window)


@register_atom(concrete_only)
def rank_peaks(residual: np.ndarray,positions: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Explicit rank peaks operation for qualified runtime composition."""
    from sciona.decomposed_vsb_features import rank_peaks as run
    return run(residual,positions)


@register_atom(concrete_only)
def amplitude_gradient(heights: np.ndarray,width: int,trim: int) -> np.ndarray:
    """Explicit amplitude gradient operation for qualified runtime composition."""
    from sciona.decomposed_vsb_features import amplitude_gradient as run
    return run(heights,width,trim)


@register_atom(concrete_only)
def select_ranked_peaks(positions: np.ndarray,heights: np.ndarray,gradient: np.ndarray,threshold: float,count: int,offset: int) -> tuple[np.ndarray,np.ndarray,int]:
    """Explicit select ranked peaks operation for qualified runtime composition."""
    from sciona.decomposed_vsb_features import select_ranked_peaks as run
    return run(positions,heights,gradient,threshold,count,offset)


@register_atom(concrete_only)
def peak_descriptors(positions: np.ndarray,residual: np.ndarray,small_radius: int,large_radius: int) -> np.ndarray:
    """Explicit peak descriptors operation for qualified runtime composition."""
    from sciona.vsb_descriptors import peak_descriptors as run
    return run(positions,residual,small_radius=small_radius,large_radius=large_radius)


@register_atom(concrete_only)
def phase_quadrants(positions: np.ndarray,coefficient: object,length: int) -> np.ndarray:
    """Explicit phase quadrants operation for qualified runtime composition."""
    from sciona.vsb_measurement import phase_quadrants as run
    return run(positions,coefficient,length=length)


@register_atom(concrete_only)
def collect(values: np.ndarray,coefficients: np.ndarray,controls: dict,arguments_function: object,filter_function: object,residual_function: object,maxima_function: object,rank_function: object,gradient_function: object,selection_function: object,descriptor_function: object,phase_function: object) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,int]:
    """Explicit collect operation for qualified runtime composition."""
    from sciona.decomposed_vsb_measurement import collect as run
    return run(values,coefficients,controls,arguments_function,filter_function,residual_function,maxima_function,rank_function,gradient_function,selection_function,descriptor_function,phase_function)


@register_atom(concrete_only)
def aggregate(groups: np.ndarray,heights: np.ndarray,descriptors: np.ndarray,quadrants: np.ndarray,measurement_count: int) -> np.ndarray:
    """Explicit aggregate operation for qualified runtime composition."""
    from sciona.vsb_measurement import aggregate as run
    return run(groups,heights,descriptors,quadrants,measurement_count=measurement_count)


@register_atom(concrete_only)
def recipe_inputs(training: np.ndarray,signal_labels: np.ndarray,query: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Explicit recipe inputs operation for qualified runtime composition."""
    from sciona.decomposed_vsb_training import recipe_inputs as run
    return run(training,signal_labels,query)


@register_atom(concrete_only)
def training_controls(controls: dict) -> tuple[int,int,int,int,int]:
    """Explicit training controls operation for qualified runtime composition."""
    return tuple(controls[k] for k in ('repetitions','split_seed','tree_seed','rounds','patience'))


@register_atom(concrete_only)
def repeated_folds(labels: np.ndarray,seed: int,repetitions: int) -> list:
    """Explicit repeated folds operation for qualified runtime composition."""
    from sciona.vsb_threshold import repeated_folds as run
    return run(labels,seed=seed,repetitions=repetitions)


@register_atom(concrete_only)
def fit_fold(training: np.ndarray,labels: np.ndarray,query: np.ndarray,train: np.ndarray,validation: np.ndarray,test: np.ndarray,tree_seed: int,rounds: int,patience: int) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,int]:
    """Explicit fit fold operation for qualified runtime composition."""
    from sciona.decomposed_vsb_training import fit_fold as run
    return run(training,labels,query,train,validation,test,tree_seed,rounds,patience)


@register_atom(concrete_only)
def ensemble(training: np.ndarray,labels: np.ndarray,query: np.ndarray,plans: list,repetitions: int,tree_seed: int,rounds: int,patience: int,fold_function: object) -> dict:
    """Explicit ensemble operation for qualified runtime composition."""
    from sciona.decomposed_vsb_training import ensemble as run
    return run(training,labels,query,plans,repetitions,tree_seed,rounds,patience,fold_function)


@register_atom(concrete_only)
def training_scores(trained: dict) -> np.ndarray:
    """Explicit training scores operation for qualified runtime composition."""
    return trained['training_probabilities']


@register_atom(concrete_only)
def threshold_inputs(signal_labels: np.ndarray,training_probabilities: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Explicit threshold inputs operation for qualified runtime composition."""
    from sciona.decomposed_vsb_training import threshold_inputs as run
    return run(signal_labels,training_probabilities)


@register_atom(concrete_only)
def threshold(labels: np.ndarray,probabilities: np.ndarray) -> dict:
    """Explicit threshold operation for qualified runtime composition."""
    from sciona.vsb_threshold import threshold as run
    return run(labels,probabilities)


@register_atom(concrete_only)
def result(ensemble_result: dict,selected: dict) -> dict:
    """Explicit result operation for qualified runtime composition."""
    from sciona.decomposed_vsb_training import result as run
    return run(ensemble_result,selected)


@register_atom(concrete_only)
def encode_result(trained: dict) -> dict:
    """Explicit encode result operation for qualified runtime composition."""
    from sciona.decomposed_vsb_contract import encode_result as run
    return run(trained)
