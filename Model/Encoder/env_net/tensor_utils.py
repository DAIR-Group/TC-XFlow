from __future__ import annotations

import torch
import torch.nn.functional as F

from Model.Encoder.env_net.feature_encoding import ENV_FEATURE_DIMS


def feat_to_tensor(feat: dict) -> torch.Tensor:

    parts = []
    for key, dim in ENV_FEATURE_DIMS.items():
        v = feat.get(key, None)
        if v is None:
            parts.append(torch.zeros(dim))
            continue
        if not isinstance(v, (list, torch.Tensor)):
            v = [float(v)]
        t = torch.tensor(v, dtype=torch.float)
        if t.numel() < dim:
            t = F.pad(t, (0, dim - t.numel()))
        parts.append(t[:dim])
    return torch.cat(parts)


def build_env_vector(env_data, B: int, T: int, device: torch.device) -> torch.Tensor:

    parts = []
    for key, dim in ENV_FEATURE_DIMS.items():
        slot = torch.zeros(B, T, dim, device=device)
        if env_data is None or key not in env_data or env_data[key] is None:
            parts.append(slot)
            continue

        v = env_data[key]
        if not torch.is_tensor(v):
            try:
                v = torch.tensor(v, dtype=torch.float, device=device)
            except Exception:
                parts.append(slot)
                continue
        v = v.float().to(device)

        try:
            if v.dim() == 0:
                slot = v.expand(B, T, 1) if dim == 1 else slot
            elif v.dim() == 1:
                if v.numel() == dim:
                    slot = v.view(1, 1, dim).expand(B, T, dim)
            elif v.dim() == 2:
                if v.shape == (B, T):
                    slot = v.unsqueeze(-1).expand(-1, -1, dim)
                elif v.shape == (B, dim):
                    slot = v.unsqueeze(1).expand(-1, T, -1)
                elif v.shape[0] == T and v.shape[1] == dim:
                    slot = v.unsqueeze(0).expand(B, -1, -1)
            elif v.dim() == 3:
                if v.shape[:2] == (B, T):
                    d_in = v.shape[-1]
                    slot = v[..., :dim] if d_in >= dim else F.pad(v, (0, dim - d_in))
                elif v.shape[0] == T:
                    s = v.permute(1, 0, 2)[..., :dim]
                    slot = s if s.shape[-1] == dim else F.pad(s, (0, dim - s.shape[-1]))
        except Exception:
            pass

        parts.append(slot.float())
    return torch.cat(parts, dim=-1)
