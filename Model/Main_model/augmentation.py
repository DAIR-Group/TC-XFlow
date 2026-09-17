
from __future__ import annotations

import math

import torch

_SHIFT_UPPER = 0.25
_SPEED_SCALE_UPPER = 0.45
_RECURVATURE_UPPER = 0.65
_NO_AUGMENT_UPPER = 0.90

def augment_batch(batch_list, disable_c: bool = False) -> list:
    batch = list(batch_list)
    if not torch.is_tensor(batch[0]):
        return batch

    obs_position = batch[0]
    device = obs_position.device
    anchor = obs_position[-1:, :, :2].detach()
    branch = torch.rand(1).item()

    if branch < _SHIFT_UPPER:
        _augment_positional_shift(batch, obs_position, device)
    elif branch < _SPEED_SCALE_UPPER:
        _augment_speed_rescale(batch, obs_position, anchor)
    elif branch < _RECURVATURE_UPPER:
        if not disable_c:
            _augment_recurvature_rotation(batch, obs_position, anchor, device)
    elif branch < _NO_AUGMENT_UPPER:
        pass  # unmodified batch
    else:
        _augment_isotropic_noise(batch, obs_position)

    return batch


def _augment_positional_shift(batch, obs_position, device):
   
    shift = (torch.rand(2, device=device) - 0.5) * 0.018
    batch[0] = obs_position + shift.view(1, 1, 2)
    if torch.is_tensor(batch[1]):
        batch[1] = batch[1] + shift.view(1, 1, 2)


def _augment_speed_rescale(batch, obs_position, anchor):
  
    scale = 0.70 + 0.70 * torch.rand(1, device=obs_position.device).item()
    obs_rescaled = obs_position.clone()
    obs_rescaled[..., :2] = anchor + (obs_position[..., :2] - anchor) * scale
    batch[0] = obs_rescaled
    if torch.is_tensor(batch[1]):
        batch[1] = anchor + (batch[1] - anchor) * scale

def _augment_recurvature_rotation(batch, obs_position, anchor, device):
   
    n_pred_steps = batch[1].shape[0] if torch.is_tensor(batch[1]) else 0
    if n_pred_steps < 4:
        return

    ground_truth = batch[1].clone()
    max_rotation_deg = (torch.rand(1).item() - 0.5) * 40.0
    max_rotation_rad = max_rotation_deg * math.pi / 180.0
    points = torch.cat([anchor, ground_truth], 0)
    displacement = points[1:] - points[:-1]

    for h in range(n_pred_steps):
        progress = (h / max(n_pred_steps - 1, 1)) ** 1.5
        angle = max_rotation_rad * progress
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=ground_truth.dtype, device=device)
        displacement[h] = (rotation @ displacement[h].unsqueeze(-1)).squeeze(-1)

    rotated_ground_truth = ground_truth.clone()
    rotated_ground_truth[0] = anchor[0] + displacement[0]
    for h in range(1, n_pred_steps):
        rotated_ground_truth[h] = rotated_ground_truth[h - 1] + displacement[h]
    batch[1] = rotated_ground_truth

    n_obs_steps = obs_position.shape[0]
    obs_rotated = obs_position.clone()
    partial_angle = max_rotation_rad * 0.3
    cos_p, sin_p = math.cos(partial_angle), math.sin(partial_angle)
    partial_rotation = torch.tensor([[cos_p, -sin_p], [sin_p, cos_p]], dtype=obs_position.dtype, device=device)
    for t in range(max(1, n_obs_steps - 3), n_obs_steps):
        step_disp = obs_rotated[t, :, :2] - obs_rotated[t - 1, :, :2]
        obs_rotated[t, :, :2] = obs_rotated[t - 1, :, :2] + (partial_rotation @ step_disp.unsqueeze(-1)).squeeze(-1)
    batch[0] = obs_rotated

def _augment_isotropic_noise(batch, obs_position):

    obs_noisy = obs_position.clone()
    obs_noisy[..., :2] = obs_position[..., :2] + torch.randn_like(obs_position[..., :2]) * 0.003
    batch[0] = obs_noisy
