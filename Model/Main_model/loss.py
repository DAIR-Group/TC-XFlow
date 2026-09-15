from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from Model.Main_model.ot_coupling import _ot_match

R_EARTH = 6371.0
DT_HOURS = 6.0


def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    return torch.stack([(t[..., 0] * 50.0 + 1800.0) / 10.0, (t[..., 1] * 50.0) / 10.0], dim=-1)


def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    lat1 = torch.deg2rad(p1[..., 1])
    lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
    a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-6, 1 - 1e-6).sqrt())


def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    lon1 = torch.deg2rad(p1[..., 0])
    lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0])
    lat2 = torch.deg2rad(p2[..., 1])
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    return torch.atan2(y, x)


def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
    if traj_deg.shape[0] < 2:
        return traj_deg.new_zeros(1, traj_deg.shape[1])
    return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


def hard_score_from_obs(
    obs_traj_norm: torch.Tensor,
    return_components: bool = False,
    weight_logits: Optional[torch.Tensor] = None,
    obs_speed_norm_const: float = 20.0,
):
    T, B = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
    device = obs_traj_norm.device
    if T < 3:
        z = torch.zeros(B, device=device)
        if return_components:
            return z, {
                "curvature": z.clone(),
                "speed_var": z.clone(),
                "dir_change": z.clone(),
                "obs_speed_norm": z.clone(),
            }
        return z

    traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
    az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
    az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
    diff = (az23 - az12).abs()
    diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
    curvature = diff.mean(0) / math.pi
    spd = _step_speeds_kmh(traj_deg)
    if spd.shape[0] >= 2:
        speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0.0, 1.0)
        obs_speed_norm = (spd.mean(0) / obs_speed_norm_const).clamp(0.0, 1.0)
    else:
        speed_var = torch.zeros(B, device=device)
        obs_speed_norm = torch.zeros(B, device=device)
    dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)

    components = torch.stack([curvature, speed_var, dir_change, obs_speed_norm], dim=0)

    if weight_logits is not None:
        w = F.softmax(weight_logits.to(device).to(components.dtype), dim=0)
    else:
        w = torch.tensor([0.35, 0.25, 0.25, 0.15], device=device, dtype=components.dtype)

    score = (w.unsqueeze(1) * components).sum(0).clamp(0.0, 1.0)

    if return_components:
        return score, {
            "curvature": curvature,
            "speed_var": speed_var,
            "dir_change": dir_change,
            "obs_speed_norm": obs_speed_norm,
        }
    return score


def _physics_score(
    traj_norm: torch.Tensor,
    obs_norm: torch.Tensor,
    use_curvature_score: bool = False,
    weight_logits: Optional[torch.Tensor] = None,
    v_sigma_scale_logit: Optional[torch.Tensor] = None,
    kernel_scale_logits: Optional[torch.Tensor] = None,
    disp_decel_logit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B = traj_norm.shape[1]
    device = traj_norm.device
    traj_deg = _norm_to_deg(traj_norm)
    v_ref = None

    if v_sigma_scale_logit is not None:
        _v_sigma_scale = torch.sigmoid(v_sigma_scale_logit.to(device))
    else:
        _v_sigma_scale = 0.5

    if kernel_scale_logits is not None:
        _ks = F.softplus(kernel_scale_logits.to(device))
        _smooth_scale, _head_scale, _disp_scale = _ks[0], _ks[1], _ks[2]
    else:
        _smooth_scale, _head_scale, _disp_scale = 5.0, 3.0, 1.5

    if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
        obs_deg = _norm_to_deg(obs_norm)
        obs_spd = _step_speeds_kmh(obs_deg)
        T_s = obs_spd.shape[0]
        w_obs = torch.linspace(0.5, 1.0, T_s, device=device)
        v_ref = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()

        v_ref = torch.nan_to_num(v_ref, nan=5.0, posinf=100.0, neginf=5.0)
        pred_spd = _step_speeds_kmh(traj_deg)
        v_sigma = v_ref.clamp(min=5.0) * _v_sigma_scale
        speed_score = torch.exp(
            -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5
        )
    elif traj_deg.shape[0] >= 2:
        speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.0).mean(0))
    else:
        speed_score = torch.ones(B, device=device)

    if traj_deg.shape[0] >= 3:
        vel = traj_deg[1:] - traj_deg[:-1]
        accel_mag = (vel[1:] - vel[:-1]).norm(dim=-1)
        smooth_score = torch.exp(-accel_mag.mean(0) * _smooth_scale)
    else:
        smooth_score = torch.ones(B, device=device)

    if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
        obs_vel = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
        pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
        obs_h = F.normalize(obs_vel, dim=-1, eps=1e-6)
        pred_h = F.normalize(pred_vel, dim=-1, eps=1e-6)
        cos_sim = (obs_h * pred_h).sum(-1).clamp(-1, 1)
        head_score = torch.exp((cos_sim - 1.0) * _head_scale)
    else:
        head_score = torch.ones(B, device=device)

    if use_curvature_score and obs_norm.shape[0] >= 3 and traj_deg.shape[0] >= 2:
        obs_deg_c = _norm_to_deg(obs_norm)
        bear_obs_1 = _forward_azimuth(obs_deg_c[-3], obs_deg_c[-2])
        bear_obs_2 = _forward_azimuth(obs_deg_c[-2], obs_deg_c[-1])
        obs_turn_rate = ((bear_obs_2 - bear_obs_1 + 180.0) % 360.0) - 180.0

        bear0 = _forward_azimuth(obs_deg_c[-1], traj_deg[0])
        if traj_deg.shape[0] >= 2:
            chain = [
                _forward_azimuth(traj_deg[t], traj_deg[t + 1]) for t in range(traj_deg.shape[0] - 1)
            ]
            pred_bears = torch.stack([bear0] + chain, 0)
        else:
            pred_bears = bear0.unsqueeze(0)

        if pred_bears.shape[0] >= 2:
            pred_turn = ((pred_bears[1:] - pred_bears[:-1] + 180.0) % 360.0) - 180.0
            turn_err = (pred_turn - obs_turn_rate.unsqueeze(0)).abs().mean(0)
            curvature_score = torch.exp(-turn_err / 15.0)
        else:
            curvature_score = torch.ones(B, device=device)
    else:
        curvature_score = torch.ones(B, device=device)

    if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
        T_pred = traj_deg.shape[0]

        if disp_decel_logit is not None:
            decel_factor = torch.sigmoid(disp_decel_logit.to(device))
        else:
            decel_factor = 0.75
        expected_total = v_ref * T_pred * DT_HOURS * decel_factor
        step_dists = _haversine_deg(traj_deg[:-1], traj_deg[1:])
        actual_total = step_dists.sum(0)

        expected_total = torch.nan_to_num(expected_total, nan=10.0, posinf=1e4, neginf=10.0)
        rel_err = (actual_total - expected_total).abs() / expected_total.clamp(min=10.0)

        rel_err = torch.nan_to_num(rel_err, nan=100.0, posinf=100.0, neginf=0.0).clamp(max=100.0)
        disp_score = torch.exp(-rel_err * _disp_scale)
    else:
        disp_score = torch.ones(B, device=device)

    _LOG_SCORE_FLOOR = -50.0

    def _safe_log(s: torch.Tensor) -> torch.Tensor:
        return torch.log(s.clamp(min=1e-30)).clamp(min=_LOG_SCORE_FLOOR)

    log_speed = _safe_log(speed_score)
    log_smooth = _safe_log(smooth_score)
    log_head = _safe_log(head_score)
    log_disp = _safe_log(disp_score)
    log_curv = _safe_log(curvature_score)

    if weight_logits is not None:
        if use_curvature_score:
            w = F.softmax(weight_logits.to(device), dim=0)
            log_combined = (
                w[0] * log_speed
                + w[1] * log_smooth
                + w[2] * log_head
                + w[3] * log_disp
                + w[4] * log_curv
            )
        else:
            w = F.softmax(weight_logits[:4].to(device), dim=0)
            log_combined = w[0] * log_speed + w[1] * log_smooth + w[2] * log_head + w[3] * log_disp
        return torch.exp(log_combined.clamp(min=_LOG_SCORE_FLOOR)).clamp(min=1e-6)

    if use_curvature_score:
        log_combined = (
            0.25 * log_speed
            + 0.20 * log_smooth
            + 0.25 * log_head
            + 0.10 * log_disp
            + 0.20 * log_curv
        )
    else:
        log_combined = 0.30 * log_speed + 0.25 * log_smooth + 0.30 * log_head + 0.15 * log_disp
    return torch.exp(log_combined.clamp(min=_LOG_SCORE_FLOOR)).clamp(min=1e-6)


def compute_obs_attribution(
    model, batch_list, device: torch.device, target_step: int = 11
) -> torch.Tensor:
    raw = _unwrap(model)
    with torch.no_grad():
        h_score = hard_score_from_obs(
            batch_list[0][:, :, :2], weight_logits=getattr(raw, "hard_score_weight_logits", None)
        )
    obs_req = batch_list[0].detach().clone().requires_grad_(True)
    bl_g = list(batch_list)
    bl_g[0] = obs_req
    with torch.enable_grad():
        cond = raw.encoder(bl_g, hard_score=h_score)
        x0 = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
        t0 = torch.zeros(obs_req.shape[1], device=device)
        v = raw.velocity(x0, t0, cond)
        pred_rel = x0 + v
        ts = min(target_step, raw.pred_len - 1)
        pred_rel[:, ts, :].norm(dim=-1).mean().backward()
    if obs_req.grad is not None:
        attr = obs_req.grad[:, :, :2].norm(dim=-1)
        attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
    else:
        attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
    return attr.detach()


@torch.no_grad()
def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
    all_deg = _norm_to_deg(all_traj)
    K, T, B = all_deg.shape[:3]
    mean_traj = all_deg.mean(0)
    std_km = torch.zeros(T, B, device=all_traj.device)
    for t in range(T):
        dists = _haversine_deg(
            all_deg[:, t].reshape(K * B, 2),
            mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2),
        ).reshape(K, B)
        std_km[t] = dists.std(0)
    s12 = min(1, T - 1)
    s72 = min(11, T - 1)
    return {
        "std_per_step": std_km,
        "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
        "mean_72h_std": float(std_km[s72].mean()),
        "mean_12h_std": float(std_km[s12].mean()),
        "high_uncertainty": std_km[s72] > 80.0,
    }


@torch.no_grad()
def compute_heading_deviation(pred_deg: torch.Tensor, gt_deg: torch.Tensor) -> torch.Tensor:
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        return pred_deg.new_zeros(1, pred_deg.shape[1])
    bear_gt = _forward_azimuth(gt_deg[: T - 1], gt_deg[1:T])
    bear_pred = _forward_azimuth(gt_deg[: T - 1], pred_deg[1:T])
    diff = (bear_pred - bear_gt).abs()
    diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
    return torch.rad2deg(diff)


@torch.no_grad()
def compute_cte_contribution(pred_deg: torch.Tensor, gt_deg: torch.Tensor) -> Dict:
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return {
            "ate_per_step": z,
            "cte_per_step": z,
            "ate_mean": z[0],
            "cte_mean": z[0],
            "ate_abs_mean": 0.0,
            "cte_abs_mean": 0.0,
        }
    bear_ref = _forward_azimuth(gt_deg[: T - 1], gt_deg[1:T])
    bear_err = _forward_azimuth(gt_deg[1:T], pred_deg[1:T])
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bear_err - bear_ref
    ate = dist_err * torch.cos(ang)
    cte = dist_err * torch.sin(ang)
    return {
        "ate_per_step": ate,
        "cte_per_step": cte,
        "ate_mean": ate.mean(0),
        "cte_mean": cte.abs().mean(0),
        "ate_abs_mean": float(ate.abs().mean()),
        "cte_abs_mean": float(cte.abs().mean()),
    }


@torch.no_grad()
def classify_hard_easy(
    obs_traj_norm, per_sample_loss=None, hard_score_p: float = 70.0, loss_p: float = 50.0
):
    scores = hard_score_from_obs(obs_traj_norm)
    B = scores.shape[0]
    if B < 4:
        return torch.zeros(B, dtype=torch.bool, device=scores.device)
    return scores >= torch.quantile(scores, hard_score_p / 100.0)


@torch.no_grad()
def classify_hard_easy_global(obs_traj_norm, global_threshold):
    return hard_score_from_obs(obs_traj_norm) >= global_threshold


@torch.no_grad()
def compute_diversity_score(candidates) -> float:
    if len(candidates) < 2:
        return 0.0
    T, B = candidates[0].shape[0], candidates[0].shape[1]
    ep_step = min(T - 1, 11)
    endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
    N = endpoints.shape[0]
    ep_mean = endpoints.mean(0, keepdim=True)
    dists = _haversine_deg(
        endpoints.reshape(N * B, 2), ep_mean.expand(N, B, 2).reshape(N * B, 2)
    ).reshape(N, B)
    return float(dists.std(0).mean())


def heading_loss_ms(model, pred_deg: torch.Tensor, obs_deg: torch.Tensor) -> torch.Tensor:

    if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
        return pred_deg.new_zeros(())

    ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])
    pts = torch.cat([obs_deg[-1:], pred_deg], 0)
    N = pred_deg.shape[0]

    ang_errs = []
    ref = ref_bear
    for t in range(N):
        pred_bear = _forward_azimuth(pts[t], pts[t + 1])
        angle_diff = pred_bear - ref
        ang_errs.append((1.0 - torch.cos(angle_diff)))
        ref = pred_bear.detach()
    ang_err_stack = torch.stack(ang_errs, dim=0)

    if model.training:
        with torch.no_grad():
            batch_mean_err = ang_err_stack.mean(dim=1)
            w = model.heading_err_ema_warmed
            old_ema = model.heading_err_ema[:N]
            blended = (
                model.heading_err_ema_decay * old_ema
                + (1.0 - model.heading_err_ema_decay) * batch_mean_err
            )
            new_ema = w * blended + (1.0 - w) * batch_mean_err
            model.heading_err_ema[:N].copy_(new_ema)
            model.heading_err_ema_warmed.fill_(1.0)

    norm = model.heading_err_ema[:N].clamp(min=0.05).detach().unsqueeze(1)
    return (ang_err_stack / norm).mean()


def reg_loss(
    model,
    x1_rel: torch.Tensor,
    last_obs: torch.Tensor,
    cond: torch.Tensor,
    hard_score: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    B, T, _ = x1_rel.shape
    device = x1_rel.device
    x0 = torch.randn_like(x1_rel) * model.sigma_inference
    t0 = torch.zeros(B, device=device)
    v = model.velocity(x0, t0, cond)
    x1_pred_abs = model._from_relative(x0 + v, last_obs).clamp(-20.0, 20.0)
    x1_gt_abs = model._from_relative(x1_rel, last_obs)
    pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))
    gt_deg = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
    dist = _haversine_deg(pred_deg, gt_deg)

    T_actual = dist.shape[0]
    if model.training:
        with torch.no_grad():
            batch_mean_dist = dist.mean(dim=1)
            w = model.reg_dist_ema_warmed
            old_ema = model.reg_dist_ema[:T_actual]
            blended = (
                model.reg_dist_ema_decay * old_ema
                + (1.0 - model.reg_dist_ema_decay) * batch_mean_dist
            )
            new_ema = w * blended + (1.0 - w) * batch_mean_dist
            model.reg_dist_ema[:T_actual].copy_(new_ema)
            model.reg_dist_ema_warmed.fill_(1.0)

    norm = model.reg_dist_ema[:T_actual].clamp(min=10.0).detach().unsqueeze(1)
    weighted_dist = dist / norm

    if hard_score is not None:
        sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)
    else:
        sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)

    REG_LOSS_REFERENCE_SCALE = 0.83
    return (weighted_dist * sw_hard).mean() * REG_LOSS_REFERENCE_SCALE


def compute_heading_loss(model, obs_traj, x1_gt, x1_rel, last_obs, cond, device, B):

    x0_h = torch.randn_like(x1_rel) * model.sigma_inference
    v_h = model.velocity(x0_h, torch.zeros(B, device=device), cond)
    x1_h_abs = model._from_relative(x0_h + v_h, last_obs).clamp(-20.0, 20.0)
    pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))
    obs_deg_h = _norm_to_deg(obs_traj[:, :, :2])
    l_heading = heading_loss_ms(model, pred_deg_h, obs_deg_h)

    gt_deg_h = _norm_to_deg(x1_gt.permute(1, 0, 2))
    pred_pts_h = torch.cat([obs_deg_h[-1:], pred_deg_h], 0)
    gt_pts_h = torch.cat([obs_deg_h[-1:], gt_deg_h], 0)
    l_anchor = x1_gt.new_zeros(())
    n_anchor_terms = 0
    for _idx in (7, 11):
        if pred_pts_h.shape[0] > _idx + 1 and gt_pts_h.shape[0] > _idx + 1:
            pred_bear_local = _forward_azimuth(pred_pts_h[_idx], pred_pts_h[_idx + 1])
            gt_bear_local = _forward_azimuth(gt_pts_h[_idx], gt_pts_h[_idx + 1])
            l_anchor = l_anchor + (1.0 - torch.cos(pred_bear_local - gt_bear_local)).mean()
            n_anchor_terms += 1
    if n_anchor_terms > 0:
        l_heading = l_heading + 0.5 * (l_anchor / n_anchor_terms)
    return l_heading


def compute_calib_loss(model, obs_traj, x1_gt, x1_rel, last_obs, cond, device, B):

    x0_c = torch.randn_like(x1_rel) * model.sigma_inference
    v_c = model.velocity(x0_c, torch.zeros(B, device=device), cond)
    pred_c_abs = model._from_relative(x0_c + v_c, last_obs).permute(1, 0, 2).clamp(-20.0, 20.0)
    cal_abs = model.speed_calibrate_pred(pred_c_abs, last_obs, obs_traj[:, :, :2])
    cal_deg = _norm_to_deg(cal_abs)
    gt_c_deg = _norm_to_deg(x1_gt.permute(1, 0, 2))
    l_calib = _haversine_deg(cal_deg, gt_c_deg).mean() / 300.0
    return torch.nan_to_num(l_calib, nan=1.0, posinf=1.0, neginf=0.0)


def compute_score_loss(model, obs_traj, x1_gt, x1_rel, last_obs, cond, device, B):

    K_score = 5
    cand_list, cand_scores = [], []
    for _ in range(K_score):
        x0k = torch.randn(B, model.pred_len, 2, device=device) * model.sigma_inference
        vk = model.velocity(x0k, torch.zeros(B, device=device), cond)
        xabsk = model._from_relative(x0k + vk, last_obs).permute(1, 0, 2).clamp(-20.0, 20.0)
        scorek = _physics_score(
            xabsk,
            obs_traj[:, :, :2],
            use_curvature_score=model.use_curvature_score_train,
            weight_logits=model.score_weight_logits,
            v_sigma_scale_logit=model.score_v_sigma_scale_logit,
            kernel_scale_logits=model.score_kernel_scale_logits,
            disp_decel_logit=model.disp_decel_logit,
        )
        cand_list.append(xabsk)
        cand_scores.append(scorek)

    cand_stack = torch.stack(cand_list, 0)
    score_stack = torch.stack(cand_scores, 0)
    w_score = F.softmax(score_stack * 3.0, dim=0)
    pred_score_avg = (cand_stack * w_score.view(K_score, 1, B, 1)).sum(0)

    pred_score_deg = _norm_to_deg(pred_score_avg)
    gt_score_deg = _norm_to_deg(x1_gt.permute(1, 0, 2))
    l_score = _haversine_deg(pred_score_deg, gt_score_deg).mean() / 300.0
    return torch.nan_to_num(l_score, nan=1.0, posinf=1.0, neginf=0.0)


def kendall_combine(
    model,
    l_cfm,
    l_reg,
    l_heading,
    l_calib,
    l_score,
    l_hard_reg,
    ramp_reg,
    ramp_dir,
    ramp_calib,
):

    HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

    prec_reg = torch.exp(-2.0 * model.log_sigma_reg.clamp(min=model.log_sigma_reg_min_clamp))
    prec_heading = torch.exp(-2.0 * model.log_sigma_heading.clamp(min=-3.0))
    prec_calib = torch.exp(-2.0 * model.log_sigma_calib.clamp(min=-3.0))
    prec_score = torch.exp(-2.0 * model.log_sigma_score.clamp(min=-3.0))

    weighted_reg = ramp_reg * (
        0.5 * prec_reg * l_reg
        + model.log_sigma_reg.clamp(min=model.log_sigma_reg_min_clamp)
        + HALF_LOG_2PI
    )
    weighted_heading = ramp_dir * (
        0.5 * prec_heading * l_heading + model.log_sigma_heading.clamp(min=-3.0) + HALF_LOG_2PI
    )
    weighted_calib = ramp_calib * (
        0.5 * prec_calib * l_calib + model.log_sigma_calib.clamp(min=-3.0) + HALF_LOG_2PI
    )
    weighted_score = ramp_calib * (
        0.5 * prec_score * l_score + model.log_sigma_score.clamp(min=-3.0) + HALF_LOG_2PI
    )

    total = (
        l_cfm
        + weighted_reg
        + weighted_heading
        + weighted_calib
        + weighted_score
        + model.lambda_hard_reg * l_hard_reg
    )
    if not torch.isfinite(total):
        total = l_cfm.new_zeros(())

    weighted = dict(
        prec_reg=prec_reg,
        prec_heading=prec_heading,
        prec_calib=prec_calib,
        prec_score=prec_score,
    )
    return total, weighted


def build_loss_dict(
    model,
    total,
    l_cfm,
    l_reg,
    l_heading,
    l_calib,
    l_score,
    l_hard_reg,
    hard_dist,
    weighted,
    ramp_reg,
    ramp_dir,
    ramp_calib,
    sigma,
    ade_log,
    h_score,
    x0,
) -> Dict:

    return {
        "total": total,
        "l_cfm": l_cfm.item(),
        "l_reg": l_reg.item() if torch.is_tensor(l_reg) else 0.0,
        "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
        "l_calib": l_calib.item(),
        "l_score": l_score.item(),
        "l_hard_reg": l_hard_reg.item(),
        "hard_dist": hard_dist.detach().tolist(),
        "lambda_hard_reg": model.lambda_hard_reg,
        "_t_l_cfm": l_cfm,
        "_t_l_reg": l_reg if torch.is_tensor(l_reg) else x0.new_zeros(()),
        "_t_l_heading": l_heading if torch.is_tensor(l_heading) else x0.new_zeros(()),
        "_t_l_calib": l_calib,
        "_t_l_score": l_score,
        "_t_l_hard_reg": l_hard_reg,
        "l_momentum": 0.0,
        "lam_reg": ramp_reg,
        "lam_dir": ramp_dir,
        "lam_calib": ramp_calib,
        "sigma": sigma,
        "ade_1step": ade_log,
        "hard_score_mean": float(h_score.detach().mean()),
        "hard_score_max": float(h_score.detach().max()),
        "learned_lambda_reg": float((0.5 * weighted["prec_reg"]).detach()),
        "learned_lambda_heading": float((0.5 * weighted["prec_heading"]).detach()),
        "learned_lambda_calib": float((0.5 * weighted["prec_calib"]).detach()),
        "learned_lambda_score": float((0.5 * weighted["prec_score"]).detach()),
        "learned_score_weights": F.softmax(model.score_weight_logits.detach(), dim=0).tolist(),
        "learned_score_v_sigma_scale": float(
            torch.sigmoid(model.score_v_sigma_scale_logit.detach())
        ),
        "learned_score_kernel_scales": F.softplus(
            model.score_kernel_scale_logits.detach()
        ).tolist(),
        "l_fm": l_cfm.item(),
        "dpe": 0.0,
        "heading": 0.0,
        "vel_reg": 0.0,
        "speed": 0.0,
        "accel": 0.0,
        "fm_mse": l_cfm.item(),
        "l_hard_total": 0.0,
        "n_hard": 0,
        "alpha_hard": 0.0,
        "l_sel_total": 0.0,
        "speed_head_l": 0.0,
        "l_speed_ratio": 0.0,
        "l_sigma_nll": 0.0,
        "learned_lambda_speed_ratio": 0.0,
        "learned_sigma_infer": float(model.sigma_inference),
    }


def get_loss_breakdown(model, batch_list, epoch: int = 0, **kwargs) -> Dict:

    obs_traj = batch_list[0]
    gt_traj = batch_list[1]
    B = obs_traj.shape[1]
    device = obs_traj.device

    sigma = model._sigma_schedule(epoch)
    x1_gt = gt_traj.permute(1, 0, 2)
    last_obs = obs_traj[-1, :, :2]
    x1_rel = model._to_relative(x1_gt, last_obs)

    h_score = hard_score_from_obs(obs_traj[:, :, :2], weight_logits=model.hard_score_weight_logits)
    cond = model.encoder(batch_list, hard_score=h_score)

    hard_dist = F.softmax(model.hard_score_weight_logits, dim=0)
    hard_uniform = hard_dist.new_full((4,), 0.25)
    l_hard_reg = torch.nan_to_num(
        ((hard_dist - hard_uniform) ** 2).sum(), nan=0.0, posinf=1.0, neginf=0.0
    )

    x0 = torch.randn_like(x1_rel) * sigma
    if model.use_ot and B >= 4:
        x0_flat, x1_flat = _ot_match(x0.reshape(B, -1), x1_rel.reshape(B, -1), model.ot_epsilon)
        x0 = x0_flat.reshape(B, model.pred_len, 2)
        x1_matched = x1_flat.reshape(B, model.pred_len, 2)
    else:
        x1_matched = x1_rel

    t = torch.rand(B, device=device)
    x_t = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
    u_target = x1_matched - x0
    v_pred = model.velocity(x_t, t, cond)
    l_cfm = F.mse_loss(v_pred, u_target)

    ramp_reg = 0.0 if epoch < 10 else (1.0 if epoch >= 30 else (epoch - 10) / 20.0)
    l_reg = reg_loss(model, x1_rel, last_obs, cond, h_score) if ramp_reg > 0.0 else x0.new_zeros(())

    ramp_dir = 0.0 if epoch < 5 else (1.0 if epoch >= 20 else (epoch - 5) / 15.0)
    if ramp_dir > 0.0:
        l_heading = compute_heading_loss(model, obs_traj, x1_gt, x1_rel, last_obs, cond, device, B)
    else:
        l_heading = x0.new_zeros(())

    ramp_calib = 0.0 if epoch < 10 else (1.0 if epoch >= 30 else (epoch - 10) / 20.0)
    l_calib = compute_calib_loss(model, obs_traj, x1_gt, x1_rel, last_obs, cond, device, B)

    l_score = compute_score_loss(model, obs_traj, x1_gt, x1_rel, last_obs, cond, device, B)

    total, weighted = kendall_combine(
        model,
        l_cfm,
        l_reg,
        l_heading,
        l_calib,
        l_score,
        l_hard_reg,
        ramp_reg,
        ramp_dir,
        ramp_calib,
    )

    with torch.no_grad():
        x0_log = torch.randn_like(x1_rel) * model.sigma_inference
        v_log = model.velocity(x0_log, torch.zeros(B, device=device), cond)
        ade_log = (
            _haversine_deg(
                _norm_to_deg(model._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
                _norm_to_deg(x1_gt.permute(1, 0, 2)),
            )
            .mean()
            .item()
        )

    return build_loss_dict(
        model,
        total,
        l_cfm,
        l_reg,
        l_heading,
        l_calib,
        l_score,
        l_hard_reg,
        hard_dist,
        weighted,
        ramp_reg,
        ramp_dir,
        ramp_calib,
        sigma,
        ade_log,
        h_score,
        x0,
    )


def get_loss(model, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
    return get_loss_breakdown(model, batch_list, epoch=epoch)["total"]
