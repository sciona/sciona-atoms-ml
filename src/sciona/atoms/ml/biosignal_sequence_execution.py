"""Draft Community boundaries for generic binary signal prediction."""
from sciona.ghost.registry import register_atom


def witness_biosignal_sequence_prepare(payload: dict) -> dict:return {'kind':'BiosignalSequence.Prepared'}


@register_atom(witness_biosignal_sequence_prepare)
def biosignal_sequence_prepare(payload: dict) -> object:
    """Validate subject-disjoint signal populations and spectral controls."""
    from sciona.biosignal_sequence_contract import prepare
    return prepare(payload)


def witness_biosignal_sequence_execute(prepared: object) -> dict:
    if prepared!={'kind':'BiosignalSequence.Prepared'}:raise ValueError('Prepared tabular input required')
    return {'kind':'BiosignalSequence.Result'}


@register_atom(witness_biosignal_sequence_execute)
def biosignal_sequence_execute(prepared: object) -> dict:
    """Train waveform/spectrogram branches, aggregate recordings and calibrate."""
    from sciona.biosignal_sequence_contract import execute
    return execute(prepared)
