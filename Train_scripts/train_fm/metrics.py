import torch

from Model.Main_model.loss import _haversine_deg

HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}


def ate_cte(pred_deg, gt_deg):
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        zero = pred_deg.new_zeros(1, pred_deg.shape[1])
        return zero, zero

    lo1 = torch.deg2rad(gt_deg[: T - 1, :, 0])
    la1 = torch.deg2rad(gt_deg[: T - 1, :, 1])
    lo2 = torch.deg2rad(gt_deg[1:T, :, 0])
    la2 = torch.deg2rad(gt_deg[1:T, :, 1])
    lo3 = torch.deg2rad(pred_deg[1:T, :, 0])
    la3 = torch.deg2rad(pred_deg[1:T, :, 1])

    y_obs = torch.sin(lo2 - lo1) * torch.cos(la2)
    x_obs = torch.cos(la1) * torch.sin(la2) - torch.sin(la1) * torch.cos(la2) * torch.cos(lo2 - lo1)
    bearing_obs = torch.atan2(y_obs, x_obs)

    y_pred = torch.sin(lo3 - lo2) * torch.cos(la3)
    x_pred = torch.cos(la2) * torch.sin(la3) - torch.sin(la2) * torch.cos(la3) * torch.cos(
        lo3 - lo2
    )
    bearing_pred = torch.atan2(y_pred, x_pred)

    total_dist = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    angle_diff = bearing_pred - bearing_obs

    along_track = total_dist * torch.cos(angle_diff)
    cross_track = total_dist * torch.sin(angle_diff)
    return along_track, cross_track
