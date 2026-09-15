import random

import numpy as np
import torch


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    return tuple(out)


def denorm_traj(n):
    r = np.zeros_like(n)
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


def to_deg(pts_01):
    return pts_01 / 10.0


def denorm_wind(wind_norm):
    return wind_norm * 25.0 + 40.0


def haversine_km(p1_deg, p2_deg):
    p1_deg = np.asarray(p1_deg)
    p2_deg = np.asarray(p2_deg)
    if p1_deg.shape != p2_deg.shape:
        raise ValueError(f"haversine_km: shape mismatch {p1_deg.shape} vs {p2_deg.shape}")
    lat1 = np.deg2rad(p1_deg[..., 1])
    lat2 = np.deg2rad(p2_deg[..., 1])
    dlat = np.deg2rad(p2_deg[..., 1] - p1_deg[..., 1])
    dlon = np.deg2rad(p2_deg[..., 0] - p1_deg[..., 0])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def extract_seq(tensor, batch_idx=0):
    t = tensor.cpu()
    if t.dim() != 3:
        raise ValueError(f"extract_seq: expected 3-D tensor, got shape {t.shape}")
    d0, d1, _ = t.shape
    if d1 == 1:
        return t[:, batch_idx, :].numpy()
    if d0 == 1:
        return t[batch_idx, :, :].numpy()
    if d0 > d1:
        return t[:, batch_idx, :].numpy()
    if d1 > d0:
        return t[batch_idx, :, :].numpy()
    return t[:, batch_idx, :].numpy()


def extract_ens(all_trajs, batch_idx=0):

    t = all_trajs.cpu()
    if t.dim() != 4:
        raise ValueError(f"extract_ens: unexpected tensor dim {t.dim()}, shape {t.shape}")
    S, d1, d2, F = t.shape
    if d1 == 1 or d2 > d1:
        return t[:, batch_idx, :, :].numpy()
    return t[:, :, batch_idx, :].numpy()
