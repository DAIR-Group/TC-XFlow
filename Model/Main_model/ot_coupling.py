from __future__ import annotations

import math
from typing import Tuple

import torch


def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:

    B = cost.shape[0]
    device = cost.device
    log_a = -math.log(B) * torch.ones(B, device=device)
    log_b = -math.log(B) * torch.ones(B, device=device)

    log_K = (-cost / epsilon).clamp(min=-50.0)
    log_u = torch.zeros(B, device=device)
    log_v = torch.zeros(B, device=device)
    for _ in range(n_iter):
        log_u = (log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)).clamp(-50.0, 50.0)
        log_v = (log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)).clamp(-50.0, 50.0)
    pi = (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)

    pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    return pi


def _ot_match(
    x0_flat: torch.Tensor, x1_flat: torch.Tensor, epsilon: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:

    B = x0_flat.shape[0]
    if B < 4:
        return x0_flat, x1_flat
    try:
        cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
        with torch.no_grad():
            pi = _sinkhorn_log(cost, epsilon=epsilon)
        flat = pi.reshape(-1).clamp(0.0)
        s = flat.sum()
        if not torch.isfinite(s) or s < 1e-10:
            return x0_flat, x1_flat
        idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
        return x0_flat[idx // B], x1_flat
    except Exception:
        return x0_flat, x1_flat
