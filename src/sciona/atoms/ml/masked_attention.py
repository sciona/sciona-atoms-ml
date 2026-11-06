"""Differentiable tensor reductions with no learned model or domain policy."""
import torch
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete floating tensors and boolean masks required')


@register_atom(concrete_only)
def masked_attention_pool(values: torch.Tensor, logits: torch.Tensor,
                          mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax-weighted reduction of values (B,T,F) over valid positions.

    Logits (B,T) share the floating dtype and device of values; mask is boolean
    (B,T) on the same device. Dimensions are nonempty, every row has a valid
    position, and valid values/logits must be finite. Masked entries may be
    nonfinite: they are replaced before arithmetic and receive zero gradient.
    Returns pooled values (B,F) and weights (B,T), preserving autograd, dtype
    and device. Inputs are unchanged. No score model, position encoding,
    training, padding generation or application-specific policy is included.
    """
    if not all(isinstance(x, torch.Tensor) for x in (values, logits, mask)):
        raise ValueError('Tensor inputs required')
    if (values.ndim != 3 or not all(values.shape) or not values.is_floating_point()
            or logits.shape != values.shape[:2] or mask.shape != logits.shape
            or logits.dtype != values.dtype or mask.dtype != torch.bool
            or logits.device != values.device or mask.device != values.device):
        raise ValueError('Aligned floating values/logits and boolean mask required')
    if not mask.any(1).all():
        raise ValueError('Each row requires a valid position')
    if not torch.isfinite(values[mask]).all() or not torch.isfinite(logits[mask]).all():
        raise ValueError('Finite valid values and logits required')
    clean = torch.where(mask[:, :, None], values, torch.zeros_like(values))
    weights = logits.masked_fill(~mask, float('-inf')).softmax(1)
    return (clean * weights[:, :, None]).sum(1), weights


@register_atom(concrete_only)
def sigmoid_context_gate(values: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Multiply a finite floating matrix by sigmoid of aligned finite logits.

    Inputs are nonempty (B,F) tensors with identical dtype/device. Returns a
    new tensor with autograd preserved and does not mutate inputs. The caller
    supplies the gate logits; no affine parameters or fitted state are owned.
    """
    if not all(isinstance(x, torch.Tensor) for x in (values, logits)):
        raise ValueError('Tensor inputs required')
    if (values.ndim != 2 or not all(values.shape) or not values.is_floating_point()
            or logits.shape != values.shape or logits.dtype != values.dtype
            or logits.device != values.device):
        raise ValueError('Aligned floating matrices required')
    if not torch.isfinite(values).all() or not torch.isfinite(logits).all():
        raise ValueError('Finite values and logits required')
    return values * torch.sigmoid(logits)
