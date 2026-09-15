from __future__ import annotations

import math

import torch


def augment_batch(batch_list, disable_c: bool = False) -> list:
    bl = list(batch_list)
    if not torch.is_tensor(bl[0]):
        return bl

    obs = bl[0]
    device = obs.device
    anchor = obs[-1:, :, :2].detach()
    r = torch.rand(1).item()

    if r < 0.25:
        _augment_shift(bl, obs, device)
    elif r < 0.45:
        _augment_speed_scale(bl, obs, anchor)
    elif r < 0.65:
        if not disable_c:
            _augment_recurvature(bl, obs, anchor, device)
    elif r < 0.90:
        pass
    else:
        _augment_noise(bl, obs)

    return bl


def _augment_shift(bl, obs, device):
    shift = (torch.rand(2, device=device) - 0.5) * 0.018
    bl[0] = obs + shift.view(1, 1, 2)
    if torch.is_tensor(bl[1]):
        bl[1] = bl[1] + shift.view(1, 1, 2)


def _augment_speed_scale(bl, obs, anchor):

    scale = 0.70 + 0.70 * torch.rand(1, device=obs.device).item()
    obs_c = obs.clone()
    obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
    bl[0] = obs_c
    if torch.is_tensor(bl[1]):
        bl[1] = anchor + (bl[1] - anchor) * scale


def _augment_recurvature(bl, obs, anchor, device):

    T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
    if T_pred < 4:
        return

    gt = bl[1].clone()
    max_deg = (torch.rand(1).item() - 0.5) * 40.0
    max_rad = max_deg * math.pi / 180.0
    pts = torch.cat([anchor, gt], 0)
    disp = pts[1:] - pts[:-1]

    for t in range(T_pred):
        progress = (t / max(T_pred - 1, 1)) ** 1.5
        a = max_rad * progress
        c, s = math.cos(a), math.sin(a)
        rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
        disp[t] = (rot @ disp[t].unsqueeze(-1)).squeeze(-1)

    gt_new = gt.clone()
    gt_new[0] = anchor[0] + disp[0]
    for t in range(1, T_pred):
        gt_new[t] = gt_new[t - 1] + disp[t]
    bl[1] = gt_new

    T_obs = obs.shape[0]
    obs_aug = obs.clone()
    cp, sp = math.cos(max_rad * 0.3), math.sin(max_rad * 0.3)
    rp = torch.tensor([[cp, -sp], [sp, cp]], dtype=obs.dtype, device=device)
    for t_obs in range(max(1, T_obs - 3), T_obs):
        d = obs_aug[t_obs, :, :2] - obs_aug[t_obs - 1, :, :2]
        obs_aug[t_obs, :, :2] = obs_aug[t_obs - 1, :, :2] + (rp @ d.unsqueeze(-1)).squeeze(-1)
    bl[0] = obs_aug


def _augment_noise(bl, obs):

    obs_new = obs.clone()
    obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
    bl[0] = obs_new
