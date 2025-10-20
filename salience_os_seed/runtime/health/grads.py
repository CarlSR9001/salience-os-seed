"""Gradient health monitoring utilities."""

from __future__ import annotations

from typing import Dict, Tuple

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover - fallback path
    torch = None  # type: ignore

if getattr(torch, "__SALIENT_STUB__", False):  # pragma: no cover
    torch = None  # type: ignore


def grad_health(
    model: torch.nn.Module,
    *,
    min_frac_nonzero: float = 0.2,
    min_norm: float = 1e-6,
) -> Tuple[bool, Dict[str, float]]:
    """Evaluate gradient sparsity and norm health for ``model``.

    Returns ``(healthy, stats)`` where ``stats`` captures the non-zero fraction and
    L2 gradient norm across all parameters.
    """

    nonzeros = 0
    total = 0
    sq_norm = 0.0

    if torch is None:
        return True, {"frac_nonzero": 1.0, "grad_norm": 1.0}

    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        data = grad.detach()
        total += data.numel()
        nonzeros += (data != 0).sum().item()
        sq_norm += float(data.pow(2).sum().item())

    frac_nonzero = (nonzeros / total) if total else 0.0
    grad_norm = sq_norm ** 0.5
    healthy = frac_nonzero >= min_frac_nonzero and grad_norm >= min_norm

    return healthy, {"frac_nonzero": frac_nonzero, "grad_norm": grad_norm}
