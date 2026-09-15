from __future__ import annotations

import numpy as np
import torch

from Model.main_model import TCFlowMatching
from Model.Data.trajectories_dataset.dataset import seq_collate

from vis.geo_utils import (
    move_batch,
    denorm_traj,
    to_deg,
    denorm_wind,
    haversine_km,
    extract_seq,
    extract_ens,
)


def load_tcxflow_checkpoint(model_path: str, device, obs_len: int = 8, pred_len: int = 12):

    ck = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = ck.get("model_cfg") or dict(pred_len=pred_len, obs_len=obs_len)
    model = TCFlowMatching(**model_cfg).to(device)
    state = ck.get("model", ck.get("model_state_dict", ck.get("model_state", ck)))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_tcxflow_inference(model, target, device, ode_steps: int = 10, num_ensemble: int = 20):

    batch = move_batch(seq_collate([target]), device)

    with torch.no_grad():
        pred_mean, pred_wind, all_trajs = model.sample(
            batch, num_ensemble=max(num_ensemble, 1), ddim_steps=ode_steps
        )

    obs_n = extract_seq(batch[0])
    gt_n = extract_seq(batch[1])
    pred_n = extract_seq(pred_mean)
    ens_n = (
        extract_ens(all_trajs)
        if (all_trajs is not None and torch.is_tensor(all_trajs) and all_trajs.dim() == 4)
        else None
    )

    wind_pred_kt = None
    if pred_wind is not None and torch.is_tensor(pred_wind):
        pred_wind_n = extract_seq(pred_wind)
        if pred_wind_n.shape[-1] >= 2:
            wind_pred_kt = denorm_wind(pred_wind_n[:, 1])

    wind_gt_kt = None
    if len(batch) > 8 and torch.is_tensor(batch[8]):
        gt_wind_n = extract_seq(batch[8])
        if gt_wind_n.shape[-1] >= 2:
            wind_gt_kt = denorm_wind(gt_wind_n[:, 1])

    is_delta = np.abs(pred_n).mean() < np.abs(obs_n).mean() * 0.15
    if is_delta:
        pred_n_abs = obs_n[-1:] + np.cumsum(pred_n, axis=0)
        ens_abs = (obs_n[-1:] + np.cumsum(ens_n, axis=1)) if ens_n is not None else None
    else:
        pred_n_abs = pred_n
        ens_abs = ens_n

    obs_deg = to_deg(denorm_traj(obs_n))
    gt_deg = to_deg(denorm_traj(gt_n))
    pred_deg = to_deg(denorm_traj(pred_n_abs))
    ens_deg = to_deg(denorm_traj(ens_abs)) if ens_abs is not None else None

    if pred_deg.shape[0] != gt_deg.shape[0]:
        T_min = min(pred_deg.shape[0], gt_deg.shape[0])
        print(
            f"LENGTH MISMATCH: pred={pred_deg.shape[0]} with gt={gt_deg.shape[0]} steps "
            f"trimming to {T_min} steps."
        )
        pred_deg = pred_deg[:T_min]
        gt_deg = gt_deg[:T_min]
        if ens_deg is not None and ens_deg.shape[1] != T_min:
            ens_deg = ens_deg[:, :T_min]
        if wind_pred_kt is not None and wind_pred_kt.shape[0] != T_min:
            wind_pred_kt = wind_pred_kt[:T_min]
        if wind_gt_kt is not None and wind_gt_kt.shape[0] != T_min:
            wind_gt_kt = wind_gt_kt[:T_min]

    errors_km = haversine_km(pred_deg, gt_deg)
    return obs_deg, gt_deg, pred_deg, ens_deg, errors_km, wind_pred_kt, wind_gt_kt
