
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from Model.Main_model.ot_coupling import resample_sources_for_targets

EARTH_RADIUS_KM = 6371.0
STEP_INTERVAL_HOURS = 6.0


def unwrap_compiled(model):
   
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def to_degrees(normalized_lonlat: torch.Tensor) -> torch.Tensor:
  
    return torch.stack(
        [
            (normalized_lonlat[..., 0] * 50.0 + 1800.0) / 10.0,
            (normalized_lonlat[..., 1] * 50.0) / 10.0,
        ],
        dim=-1,
    )


def haversine_km(p1_deg: torch.Tensor, p2_deg: torch.Tensor) -> torch.Tensor:
   
    lat1 = torch.deg2rad(p1_deg[..., 1])
    lat2 = torch.deg2rad(p2_deg[..., 1])
    dlat = torch.deg2rad(p2_deg[..., 1] - p1_deg[..., 1])
    dlon = torch.deg2rad(p2_deg[..., 0] - p1_deg[..., 0])
    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
    a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return 2.0 * EARTH_RADIUS_KM * torch.asin(a.clamp(1e-6, 1 - 1e-6).sqrt())


def forward_bearing(p1_deg: torch.Tensor, p2_deg: torch.Tensor) -> torch.Tensor:

    lon1 = torch.deg2rad(p1_deg[..., 0])
    lat1 = torch.deg2rad(p1_deg[..., 1])
    lon2 = torch.deg2rad(p2_deg[..., 0])
    lat2 = torch.deg2rad(p2_deg[..., 1])
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    return torch.atan2(y, x)


def step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
    
    if traj_deg.shape[0] < 2:
        return traj_deg.new_zeros(1, traj_deg.shape[1])
    return haversine_km(traj_deg[:-1], traj_deg[1:]) / STEP_INTERVAL_HOURS


def difficulty_score_from_history(
    obs_traj_norm: torch.Tensor,
    return_components: bool = False,
    weight_logits: Optional[torch.Tensor] = None,
    obs_speed_norm_const: float = 20.0,
):
   
    n_obs_steps, batch_size = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
    device = obs_traj_norm.device
    if n_obs_steps < 3:
        zero = torch.zeros(batch_size, device=device)
        if return_components:
            return zero, {
                "turning_magnitude": zero.clone(),
                "speed_variability": zero.clone(),
                "turning_frequency": zero.clone(),
                "mean_translation_speed": zero.clone(),
            }
        return zero

    traj_deg = to_degrees(obs_traj_norm[..., :2])
    bearing_step1 = forward_bearing(traj_deg[:-2], traj_deg[1:-1])
    bearing_step2 = forward_bearing(traj_deg[1:-1], traj_deg[2:])
    turning_angle = (bearing_step2 - bearing_step1).abs()
    turning_angle = torch.where(turning_angle > math.pi, 2 * math.pi - turning_angle, turning_angle)

    turning_magnitude = turning_angle.mean(0) / math.pi  # comp_1, Eq. comp1

    speed = step_speeds_kmh(traj_deg)
    if speed.shape[0] >= 2:
        speed_variability = (speed.std(0) / speed.mean(0).clamp(min=1.0)).clamp(0.0, 1.0)  # comp_2, Eq. comp2
        mean_translation_speed = (speed.mean(0) / obs_speed_norm_const).clamp(0.0, 1.0)  # comp_4, Eq. comp4
    else:
        speed_variability = torch.zeros(batch_size, device=device)
        mean_translation_speed = torch.zeros(batch_size, device=device)

    turning_frequency = (turning_angle > (20.0 / 180.0 * math.pi)).float().mean(0)  # comp_3, Eq. comp3

    components = torch.stack(
        [turning_magnitude, speed_variability, turning_frequency, mean_translation_speed], dim=0
    )

    if weight_logits is not None:
        mixture_weights = F.softmax(weight_logits.to(device).to(components.dtype), dim=0)
    else:
        mixture_weights = torch.tensor([0.35, 0.25, 0.25, 0.15], device=device, dtype=components.dtype)

    rho = (mixture_weights.unsqueeze(1) * components).sum(0).clamp(0.0, 1.0)

    if return_components:
        return rho, {
            "turning_magnitude": turning_magnitude,
            "speed_variability": speed_variability,
            "turning_frequency": turning_frequency,
            "mean_translation_speed": mean_translation_speed,
        }
    return rho


def physics_plausibility_score(
    candidate_traj_deg: torch.Tensor,
    obs_traj_deg: torch.Tensor,
    use_turning_rate_criterion: bool = False,
    weight_logits: Optional[torch.Tensor] = None,
    speed_sigma_scale_logit: Optional[torch.Tensor] = None,
    kernel_scale_logits: Optional[torch.Tensor] = None,
    displacement_decel_logit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
   
    batch_size = candidate_traj_deg.shape[1]
    device = candidate_traj_deg.device
    reference_speed = None

    if speed_sigma_scale_logit is not None:
        speed_sigma_scale = torch.sigmoid(speed_sigma_scale_logit.to(device))
    else:
        speed_sigma_scale = 0.5

    if kernel_scale_logits is not None:
        kernel_scales = F.softplus(kernel_scale_logits.to(device))
        smoothness_scale, heading_scale, displacement_scale = (
            kernel_scales[0],
            kernel_scales[1],
            kernel_scales[2],
        )
    else:
        smoothness_scale, heading_scale, displacement_scale = 5.0, 3.0, 1.5

    # Speed score
    if candidate_traj_deg.shape[0] >= 2 and obs_traj_deg.shape[0] >= 2:
        obs_speed = step_speeds_kmh(obs_traj_deg)
        n_obs_speed_steps = obs_speed.shape[0]
        recency_weights = torch.linspace(0.5, 1.0, n_obs_speed_steps, device=device)
        reference_speed = (obs_speed * recency_weights.unsqueeze(1)).sum(0) / recency_weights.sum()
        reference_speed = torch.nan_to_num(reference_speed, nan=5.0, posinf=100.0, neginf=5.0)

        candidate_speed = step_speeds_kmh(candidate_traj_deg)
        speed_sigma = reference_speed.clamp(min=5.0) * speed_sigma_scale
        speed_score = torch.exp(
            -((candidate_speed - reference_speed.unsqueeze(0)) / speed_sigma.unsqueeze(0)).pow(2).mean(0)
            * 0.5
        )
    elif candidate_traj_deg.shape[0] >= 2:
        speed_score = torch.exp(-(step_speeds_kmh(candidate_traj_deg).clamp(min=0) / 30.0).mean(0))
    else:
        speed_score = torch.ones(batch_size, device=device)

    # Smoothness score:
    if candidate_traj_deg.shape[0] >= 3:
        velocity = candidate_traj_deg[1:] - candidate_traj_deg[:-1]
        acceleration_mag = (velocity[1:] - velocity[:-1]).norm(dim=-1)
        smoothness_score = torch.exp(-acceleration_mag.mean(0) * smoothness_scale)
    else:
        smoothness_score = torch.ones(batch_size, device=device)

    # Heading score
    if obs_traj_deg.shape[0] >= 2 and candidate_traj_deg.shape[0] >= 1:
        obs_velocity = obs_traj_deg[-1, :, :2] - obs_traj_deg[-2, :, :2]
        candidate_first_velocity = candidate_traj_deg[0, :, :2] - obs_traj_deg[-1, :, :2]
        obs_heading_unit = F.normalize(obs_velocity, dim=-1, eps=1e-6)
        candidate_heading_unit = F.normalize(candidate_first_velocity, dim=-1, eps=1e-6)
        cosine_similarity = (obs_heading_unit * candidate_heading_unit).sum(-1).clamp(-1, 1)
        heading_score = torch.exp((cosine_similarity - 1.0) * heading_scale)
    else:
        heading_score = torch.ones(batch_size, device=device)

    # Turning-rate score
    if use_turning_rate_criterion and obs_traj_deg.shape[0] >= 3 and candidate_traj_deg.shape[0] >= 2:
        obs_bearing_1 = forward_bearing(obs_traj_deg[-3], obs_traj_deg[-2])
        obs_bearing_2 = forward_bearing(obs_traj_deg[-2], obs_traj_deg[-1])
        obs_turn_rate = ((obs_bearing_2 - obs_bearing_1 + 180.0) % 360.0) - 180.0

        first_candidate_bearing = forward_bearing(obs_traj_deg[-1], candidate_traj_deg[0])
        if candidate_traj_deg.shape[0] >= 2:
            chained_bearings = [
                forward_bearing(candidate_traj_deg[t], candidate_traj_deg[t + 1])
                for t in range(candidate_traj_deg.shape[0] - 1)
            ]
            candidate_bearings = torch.stack([first_candidate_bearing] + chained_bearings, 0)
        else:
            candidate_bearings = first_candidate_bearing.unsqueeze(0)

        if candidate_bearings.shape[0] >= 2:
            candidate_turn_rate = ((candidate_bearings[1:] - candidate_bearings[:-1] + 180.0) % 360.0) - 180.0
            turn_rate_error = (candidate_turn_rate - obs_turn_rate.unsqueeze(0)).abs().mean(0)
            turning_rate_score = torch.exp(-turn_rate_error / 15.0)
        else:
            turning_rate_score = torch.ones(batch_size, device=device)
    else:
        turning_rate_score = torch.ones(batch_size, device=device)

    # Displacement score
    if reference_speed is not None and candidate_traj_deg.shape[0] >= 2 and obs_traj_deg.shape[0] >= 2:
        n_pred_steps = candidate_traj_deg.shape[0]

        if displacement_decel_logit is not None:
            deceleration_factor = torch.sigmoid(displacement_decel_logit.to(device))
        else:
            deceleration_factor = 0.75
        expected_total_displacement = (
            reference_speed * n_pred_steps * STEP_INTERVAL_HOURS * deceleration_factor
        )
        step_distances = haversine_km(candidate_traj_deg[:-1], candidate_traj_deg[1:])
        actual_total_displacement = step_distances.sum(0)

        expected_total_displacement = torch.nan_to_num(
            expected_total_displacement, nan=10.0, posinf=1e4, neginf=10.0
        )
        relative_error = (actual_total_displacement - expected_total_displacement).abs() / (
            expected_total_displacement.clamp(min=10.0)
        )
        relative_error = torch.nan_to_num(relative_error, nan=100.0, posinf=100.0, neginf=0.0).clamp(
            max=100.0
        )
        displacement_score = torch.exp(-relative_error * displacement_scale)
    else:
        displacement_score = torch.ones(batch_size, device=device)

    _LOG_SCORE_FLOOR = -50.0

    def _safe_log(score: torch.Tensor) -> torch.Tensor:
        return torch.log(score.clamp(min=1e-30)).clamp(min=_LOG_SCORE_FLOOR)

    log_speed = _safe_log(speed_score)
    log_smoothness = _safe_log(smoothness_score)
    log_heading = _safe_log(heading_score)
    log_displacement = _safe_log(displacement_score)
    log_turning_rate = _safe_log(turning_rate_score)

    if weight_logits is not None:
        if use_turning_rate_criterion:
            weights = F.softmax(weight_logits.to(device), dim=0)
            log_combined = (
                weights[0] * log_speed
                + weights[1] * log_smoothness
                + weights[2] * log_heading
                + weights[3] * log_displacement
                + weights[4] * log_turning_rate
            )
        else:
            weights = F.softmax(weight_logits[:4].to(device), dim=0)
            log_combined = (
                weights[0] * log_speed
                + weights[1] * log_smoothness
                + weights[2] * log_heading
                + weights[3] * log_displacement
            )
        return torch.exp(log_combined.clamp(min=_LOG_SCORE_FLOOR)).clamp(min=1e-6)

    if use_turning_rate_criterion:
        log_combined = (
            0.25 * log_speed
            + 0.20 * log_smoothness
            + 0.25 * log_heading
            + 0.10 * log_displacement
            + 0.20 * log_turning_rate
        )
    else:
        log_combined = 0.30 * log_speed + 0.25 * log_smoothness + 0.30 * log_heading + 0.15 * log_displacement
    return torch.exp(log_combined.clamp(min=_LOG_SCORE_FLOOR)).clamp(min=1e-6)



def compute_context_attribution(
    model, batch_list, device: torch.device, target_horizon_step: int = 11
) -> torch.Tensor:
    
    raw_model = unwrap_compiled(model)
    with torch.no_grad():
        rho = difficulty_score_from_history(
            batch_list[0][:, :, :2], weight_logits=getattr(raw_model, "hard_score_weight_logits", None)
        )
    obs_traj_grad = batch_list[0].detach().clone().requires_grad_(True)
    batch_with_grad = list(batch_list)
    batch_with_grad[0] = obs_traj_grad
    with torch.enable_grad():
        cond = raw_model.encoder(batch_with_grad, hard_score=rho)
        x0 = torch.randn(obs_traj_grad.shape[1], raw_model.pred_len, 2, device=device) * raw_model.sigma_inference
        t0 = torch.zeros(obs_traj_grad.shape[1], device=device)
        velocity = raw_model.velocity(x0, t0, cond)
        pred_rel = x0 + velocity
        step = min(target_horizon_step, raw_model.pred_len - 1)
        pred_rel[:, step, :].norm(dim=-1).mean().backward()
    if obs_traj_grad.grad is not None:
        attribution = obs_traj_grad.grad[:, :, :2].norm(dim=-1)
        attribution = attribution / (attribution.sum(0, keepdim=True) + 1e-8)
    else:
        attribution = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
    return attribution.detach()


@torch.no_grad()
def compute_ensemble_uncertainty(all_candidates_deg: torch.Tensor) -> Dict:

    all_deg = to_degrees(all_candidates_deg)
    n_candidates, n_horizons, batch_size = all_deg.shape[:3]
    mean_traj = all_deg.mean(0)
    std_km = torch.zeros(n_horizons, batch_size, device=all_candidates_deg.device)
    for h in range(n_horizons):
        dists = haversine_km(
            all_deg[:, h].reshape(n_candidates * batch_size, 2),
            mean_traj[h].unsqueeze(0).expand(n_candidates, batch_size, 2).reshape(n_candidates * batch_size, 2),
        ).reshape(n_candidates, batch_size)
        std_km[h] = dists.std(0)
    step_12h = min(1, n_horizons - 1)
    step_72h = min(11, n_horizons - 1)
    return {
        "std_per_step": std_km,
        "uncertainty_ratio": (std_km[step_72h] + 1e-3) / (std_km[step_12h] + 1e-3),
        "mean_72h_std": float(std_km[step_72h].mean()),
        "mean_12h_std": float(std_km[step_12h].mean()),
        "high_uncertainty": std_km[step_72h] > 80.0,
    }


@torch.no_grad()
def compute_heading_deviation(pred_deg: torch.Tensor, gt_deg: torch.Tensor) -> torch.Tensor:
   
    n_steps = min(pred_deg.shape[0], gt_deg.shape[0])
    if n_steps < 2:
        return pred_deg.new_zeros(1, pred_deg.shape[1])
    gt_bearing = forward_bearing(gt_deg[: n_steps - 1], gt_deg[1:n_steps])
    pred_bearing = forward_bearing(gt_deg[: n_steps - 1], pred_deg[1:n_steps])
    deviation = (pred_bearing - gt_bearing).abs()
    deviation = torch.where(deviation > math.pi, 2 * math.pi - deviation, deviation)
    return torch.rad2deg(deviation)


@torch.no_grad()
def compute_ate_cte_decomposition(pred_deg: torch.Tensor, gt_deg: torch.Tensor) -> Dict:

    n_steps = min(pred_deg.shape[0], gt_deg.shape[0])
    if n_steps < 2:
        zero = pred_deg.new_zeros(1, pred_deg.shape[1])
        return {
            "ate_per_step": zero,
            "cte_per_step": zero,
            "ate_mean": zero[0],
            "cte_mean": zero[0],
            "ate_abs_mean": 0.0,
            "cte_abs_mean": 0.0,
        }
    reference_bearing = forward_bearing(gt_deg[: n_steps - 1], gt_deg[1:n_steps])
    error_bearing = forward_bearing(gt_deg[1:n_steps], pred_deg[1:n_steps])
    error_distance = haversine_km(pred_deg[1:n_steps], gt_deg[1:n_steps])
    bearing_offset = error_bearing - reference_bearing
    along_track_error = error_distance * torch.cos(bearing_offset)
    cross_track_error = error_distance * torch.sin(bearing_offset)
    return {
        "ate_per_step": along_track_error,
        "cte_per_step": cross_track_error,
        "ate_mean": along_track_error.mean(0),
        "cte_mean": cross_track_error.abs().mean(0),
        "ate_abs_mean": float(along_track_error.abs().mean()),
        "cte_abs_mean": float(cross_track_error.abs().mean()),
    }


@torch.no_grad()
def classify_hard_easy(
    obs_traj_norm, per_sample_loss=None, hard_score_percentile: float = 70.0, loss_percentile: float = 50.0
):

    scores = difficulty_score_from_history(obs_traj_norm)
    batch_size = scores.shape[0]
    if batch_size < 4:
        return torch.zeros(batch_size, dtype=torch.bool, device=scores.device)
    return scores >= torch.quantile(scores, hard_score_percentile / 100.0)


@torch.no_grad()
def classify_hard_easy_global(obs_traj_norm, global_threshold):
    return difficulty_score_from_history(obs_traj_norm) >= global_threshold


@torch.no_grad()
def compute_diversity_score(candidates) -> float:

    if len(candidates) < 2:
        return 0.0
    n_horizons, batch_size = candidates[0].shape[0], candidates[0].shape[1]
    endpoint_step = min(n_horizons - 1, 11)
    endpoints = torch.stack([to_degrees(c[endpoint_step]) for c in candidates], 0)
    n_candidates = endpoints.shape[0]
    endpoint_mean = endpoints.mean(0, keepdim=True)
    dists = haversine_km(
        endpoints.reshape(n_candidates * batch_size, 2),
        endpoint_mean.expand(n_candidates, batch_size, 2).reshape(n_candidates * batch_size, 2),
    ).reshape(n_candidates, batch_size)
    return float(dists.std(0).mean())


def heading_consistency_loss(model, pred_deg: torch.Tensor, obs_deg: torch.Tensor) -> torch.Tensor:

    if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
        return pred_deg.new_zeros(())

    reference_bearing = forward_bearing(obs_deg[-2], obs_deg[-1])
    all_points = torch.cat([obs_deg[-1:], pred_deg], 0)
    n_pred_steps = pred_deg.shape[0]

    angular_errors = []
    current_reference = reference_bearing
    for h in range(n_pred_steps):
        predicted_bearing = forward_bearing(all_points[h], all_points[h + 1])
        angle_diff = predicted_bearing - current_reference
        angular_errors.append(1.0 - torch.cos(angle_diff))
        current_reference = predicted_bearing.detach()
    angular_error_per_horizon = torch.stack(angular_errors, dim=0)

    if model.training:
        with torch.no_grad():
            batch_mean_error = angular_error_per_horizon.mean(dim=1)
            warmup_weight = model.heading_err_ema_warmed
            previous_ema = model.heading_err_ema[:n_pred_steps]
            blended_ema = (
                model.heading_err_ema_decay * previous_ema
                + (1.0 - model.heading_err_ema_decay) * batch_mean_error
            )
            updated_ema = warmup_weight * blended_ema + (1.0 - warmup_weight) * batch_mean_error
            model.heading_err_ema[:n_pred_steps].copy_(updated_ema)
            model.heading_err_ema_warmed.fill_(1.0)

    normalizer = model.heading_err_ema[:n_pred_steps].clamp(min=0.05).detach().unsqueeze(1)
    return (angular_error_per_horizon / normalizer).mean()


def displacement_regression_loss(
    model,
    target_displacement_rel: torch.Tensor,
    last_obs: torch.Tensor,
    context: torch.Tensor,
    difficulty_score: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    batch_size, n_pred_steps, _ = target_displacement_rel.shape
    device = target_displacement_rel.device
    x0 = torch.randn_like(target_displacement_rel) * model.sigma_inference
    t0 = torch.zeros(batch_size, device=device)
    velocity = model.velocity(x0, t0, context)
    pred_abs = model._from_relative(x0 + velocity, last_obs).clamp(-20.0, 20.0)
    gt_abs = model._from_relative(target_displacement_rel, last_obs)
    pred_deg = to_degrees(pred_abs.permute(1, 0, 2))
    gt_deg = to_degrees(gt_abs.permute(1, 0, 2))
    per_horizon_error = haversine_km(pred_deg, gt_deg)

    n_horizons_actual = per_horizon_error.shape[0]
    if model.training:
        with torch.no_grad():
            batch_mean_error = per_horizon_error.mean(dim=1)
            warmup_weight = model.reg_dist_ema_warmed
            previous_ema = model.reg_dist_ema[:n_horizons_actual]
            blended_ema = (
                model.reg_dist_ema_decay * previous_ema
                + (1.0 - model.reg_dist_ema_decay) * batch_mean_error
            )
            updated_ema = warmup_weight * blended_ema + (1.0 - warmup_weight) * batch_mean_error
            model.reg_dist_ema[:n_horizons_actual].copy_(updated_ema)
            model.reg_dist_ema_warmed.fill_(1.0)

    normalizer = model.reg_dist_ema[:n_horizons_actual].clamp(min=10.0).detach().unsqueeze(1)
    normalized_error = per_horizon_error / normalizer

    if difficulty_score is not None:
        difficulty_weight = (1.0 + difficulty_score.to(device).to(per_horizon_error.dtype)).unsqueeze(0)
    else:
        difficulty_weight = torch.ones(1, batch_size, device=device, dtype=per_horizon_error.dtype)

    REG_LOSS_REFERENCE_SCALE = 0.83 
    return (normalized_error * difficulty_weight).mean() * REG_LOSS_REFERENCE_SCALE


def heading_loss_with_anchor(model, obs_traj, gt_traj_abs, target_displacement_rel, last_obs, context, device, batch_size):
   

    x0 = torch.randn_like(target_displacement_rel) * model.sigma_inference
    velocity = model.velocity(x0, torch.zeros(batch_size, device=device), context)
    pred_abs = model._from_relative(x0 + velocity, last_obs).clamp(-20.0, 20.0)
    pred_deg = to_degrees(pred_abs.permute(1, 0, 2))
    obs_deg = to_degrees(obs_traj[:, :, :2])
    l_heading = heading_consistency_loss(model, pred_deg, obs_deg)

    gt_deg = to_degrees(gt_traj_abs.permute(1, 0, 2))
    pred_points = torch.cat([obs_deg[-1:], pred_deg], 0)
    gt_points = torch.cat([obs_deg[-1:], gt_deg], 0)
    ANCHOR_WEIGHT = 0.5
    ANCHOR_HORIZON_INDICES = (7, 11)  

    anchor_loss = gt_traj_abs.new_zeros(())
    n_anchor_terms = 0
    for horizon_idx in ANCHOR_HORIZON_INDICES:
        if pred_points.shape[0] > horizon_idx + 1 and gt_points.shape[0] > horizon_idx + 1:
            pred_bearing_local = forward_bearing(pred_points[horizon_idx], pred_points[horizon_idx + 1])
            gt_bearing_local = forward_bearing(gt_points[horizon_idx], gt_points[horizon_idx + 1])
            anchor_loss = anchor_loss + (1.0 - torch.cos(pred_bearing_local - gt_bearing_local)).mean()
            n_anchor_terms += 1
    if n_anchor_terms > 0:
        l_heading = l_heading + ANCHOR_WEIGHT * (anchor_loss / n_anchor_terms)
    return l_heading


def speed_calibration_loss(model, obs_traj, gt_traj_abs, target_displacement_rel, last_obs, context, device, batch_size):
  

    x0 = torch.randn_like(target_displacement_rel) * model.sigma_inference
    velocity = model.velocity(x0, torch.zeros(batch_size, device=device), context)
    pred_abs = model._from_relative(x0 + velocity, last_obs).permute(1, 0, 2).clamp(-20.0, 20.0)
    calibrated_abs = model.speed_calibrate_pred(pred_abs, last_obs, obs_traj[:, :, :2])
    calibrated_deg = to_degrees(calibrated_abs)
    gt_deg = to_degrees(gt_traj_abs.permute(1, 0, 2))
    CALIB_REFERENCE_SCALE_KM = 300.0
    l_calib = haversine_km(calibrated_deg, gt_deg).mean() / CALIB_REFERENCE_SCALE_KM
    return torch.nan_to_num(l_calib, nan=1.0, posinf=1.0, neginf=0.0)


def candidate_selection_loss(model, obs_traj, gt_traj_abs, target_displacement_rel, last_obs, context, device, batch_size):

    N_TRAINING_CANDIDATES = 5  
    SCORE_SOFTMAX_SHARPNESS = 3.0 
    SCORE_REFERENCE_SCALE_KM = 300.0

    candidate_trajectories, candidate_scores = [], []
    for _ in range(N_TRAINING_CANDIDATES):
        x0_k = torch.randn(batch_size, model.pred_len, 2, device=device) * model.sigma_inference
        velocity_k = model.velocity(x0_k, torch.zeros(batch_size, device=device), context)
        candidate_abs_k = model._from_relative(x0_k + velocity_k, last_obs).permute(1, 0, 2).clamp(-20.0, 20.0)
        score_k = physics_plausibility_score(
            candidate_abs_k,
            obs_traj[:, :, :2],
            use_turning_rate_criterion=model.use_curvature_score_train,
            weight_logits=model.score_weight_logits,
            speed_sigma_scale_logit=model.score_v_sigma_scale_logit,
            kernel_scale_logits=model.score_kernel_scale_logits,
            displacement_decel_logit=model.disp_decel_logit,
        )
        candidate_trajectories.append(candidate_abs_k)
        candidate_scores.append(score_k)

    stacked_candidates = torch.stack(candidate_trajectories, 0)
    stacked_scores = torch.stack(candidate_scores, 0)
    blend_weights = F.softmax(stacked_scores * SCORE_SOFTMAX_SHARPNESS, dim=0)
    blended_prediction = (stacked_candidates * blend_weights.view(N_TRAINING_CANDIDATES, 1, batch_size, 1)).sum(0)

    blended_deg = to_degrees(blended_prediction)
    gt_deg = to_degrees(gt_traj_abs.permute(1, 0, 2))
    l_score = haversine_km(blended_deg, gt_deg).mean() / SCORE_REFERENCE_SCALE_KM
    return torch.nan_to_num(l_score, nan=1.0, posinf=1.0, neginf=0.0)


def combine_losses_kendall(
    model,
    l_cfm,
    l_reg,
    l_heading,
    l_calib,
    l_score,
    l_hard_reg,
    ramp_reg,
    ramp_heading,
    ramp_calib,
):
   
    HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

    precision_reg = torch.exp(-2.0 * model.log_sigma_reg.clamp(min=model.log_sigma_reg_min_clamp))
    precision_heading = torch.exp(-2.0 * model.log_sigma_heading.clamp(min=-3.0))
    precision_calib = torch.exp(-2.0 * model.log_sigma_calib.clamp(min=-3.0))
    precision_score = torch.exp(-2.0 * model.log_sigma_score.clamp(min=-3.0))

    weighted_reg = ramp_reg * (
        0.5 * precision_reg * l_reg
        + model.log_sigma_reg.clamp(min=model.log_sigma_reg_min_clamp)
        + HALF_LOG_2PI
    )
    weighted_heading = ramp_heading * (
        0.5 * precision_heading * l_heading + model.log_sigma_heading.clamp(min=-3.0) + HALF_LOG_2PI
    )
    weighted_calib = ramp_calib * (
        0.5 * precision_calib * l_calib + model.log_sigma_calib.clamp(min=-3.0) + HALF_LOG_2PI
    )
    weighted_score = ramp_calib * (
        0.5 * precision_score * l_score + model.log_sigma_score.clamp(min=-3.0) + HALF_LOG_2PI
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

    precisions = dict(
        prec_reg=precision_reg,
        prec_heading=precision_heading,
        prec_calib=precision_calib,
        prec_score=precision_score,
    )
    return total, precisions


def build_loss_report(
    model,
    total,
    l_cfm,
    l_reg,
    l_heading,
    l_calib,
    l_score,
    l_hard_reg,
    hard_score_mixture,
    precisions,
    ramp_reg,
    ramp_heading,
    ramp_calib,
    sigma,
    one_step_ade_km,
    difficulty_score,
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
        "hard_dist": hard_score_mixture.detach().tolist(),
        "lambda_hard_reg": model.lambda_hard_reg,
        "_t_l_cfm": l_cfm,
        "_t_l_reg": l_reg if torch.is_tensor(l_reg) else x0.new_zeros(()),
        "_t_l_heading": l_heading if torch.is_tensor(l_heading) else x0.new_zeros(()),
        "_t_l_calib": l_calib,
        "_t_l_score": l_score,
        "_t_l_hard_reg": l_hard_reg,
        "lam_reg": ramp_reg,
        "lam_dir": ramp_heading,
        "lam_calib": ramp_calib,
        "sigma": sigma,
        "ade_1step": one_step_ade_km,
        "hard_score_mean": float(difficulty_score.detach().mean()),
        "hard_score_max": float(difficulty_score.detach().max()),
        "learned_lambda_reg": float((0.5 * precisions["prec_reg"]).detach()),
        "learned_lambda_heading": float((0.5 * precisions["prec_heading"]).detach()),
        "learned_lambda_calib": float((0.5 * precisions["prec_calib"]).detach()),
        "learned_lambda_score": float((0.5 * precisions["prec_score"]).detach()),
        "learned_score_weights": F.softmax(model.score_weight_logits.detach(), dim=0).tolist(),
        "learned_score_v_sigma_scale": float(torch.sigmoid(model.score_v_sigma_scale_logit.detach())),
        "learned_score_kernel_scales": F.softplus(model.score_kernel_scale_logits.detach()).tolist(),
        "learned_sigma_infer": float(model.sigma_inference),
    }


def get_loss_breakdown(model, batch_list, epoch: int = 0, **kwargs) -> Dict:

    obs_traj = batch_list[0]
    gt_traj = batch_list[1]
    batch_size = obs_traj.shape[1]
    device = obs_traj.device

    sigma = model._sigma_schedule(epoch)
    gt_traj_abs = gt_traj.permute(1, 0, 2)
    last_obs = obs_traj[-1, :, :2]
    target_displacement_rel = model._to_relative(gt_traj_abs, last_obs)

    difficulty_score = difficulty_score_from_history(
        obs_traj[:, :, :2], weight_logits=model.hard_score_weight_logits
    )
    context = model.encoder(batch_list, hard_score=difficulty_score)

    hard_score_mixture = F.softmax(model.hard_score_weight_logits, dim=0)
    uniform_mixture = hard_score_mixture.new_full((4,), 0.25)
    l_hard_reg = torch.nan_to_num(
        ((hard_score_mixture - uniform_mixture) ** 2).sum(), nan=0.0, posinf=1.0, neginf=0.0
    )

    # Conditional Flow Matching loss
    x0 = torch.randn_like(target_displacement_rel) * sigma
    if model.use_ot and batch_size >= 4:
        x0_flat, x1_flat = resample_sources_for_targets(
            x0.reshape(batch_size, -1), target_displacement_rel.reshape(batch_size, -1), model.ot_epsilon
        )
        x0 = x0_flat.reshape(batch_size, model.pred_len, 2)
        matched_target = x1_flat.reshape(batch_size, model.pred_len, 2)
    else:
        matched_target = target_displacement_rel

    tau = torch.rand(batch_size, device=device)
    z_tau = (1.0 - tau.view(batch_size, 1, 1)) * x0 + tau.view(batch_size, 1, 1) * matched_target
    target_velocity = matched_target - x0
    predicted_velocity = model.velocity(z_tau, tau, context)
    l_cfm = F.mse_loss(predicted_velocity, target_velocity)


    ramp_reg = 0.0 if epoch < 10 else (1.0 if epoch >= 30 else (epoch - 10) / 20.0)
    l_reg = (
        displacement_regression_loss(model, target_displacement_rel, last_obs, context, difficulty_score)
        if ramp_reg > 0.0
        else x0.new_zeros(())
    )

    ramp_heading = 0.0 if epoch < 5 else (1.0 if epoch >= 20 else (epoch - 5) / 15.0)
    if ramp_heading > 0.0:
        l_heading = heading_loss_with_anchor(
            model, obs_traj, gt_traj_abs, target_displacement_rel, last_obs, context, device, batch_size
        )
    else:
        l_heading = x0.new_zeros(())

    ramp_calib = 0.0 if epoch < 10 else (1.0 if epoch >= 30 else (epoch - 10) / 20.0)
    l_calib = speed_calibration_loss(
        model, obs_traj, gt_traj_abs, target_displacement_rel, last_obs, context, device, batch_size
    )

    l_score = candidate_selection_loss(
        model, obs_traj, gt_traj_abs, target_displacement_rel, last_obs, context, device, batch_size
    )

    total, precisions = combine_losses_kendall(
        model, l_cfm, l_reg, l_heading, l_calib, l_score, l_hard_reg, ramp_reg, ramp_heading, ramp_calib
    )

    with torch.no_grad():
        x0_log = torch.randn_like(target_displacement_rel) * model.sigma_inference
        velocity_log = model.velocity(x0_log, torch.zeros(batch_size, device=device), context)
        one_step_ade_km = (
            haversine_km(
                to_degrees(model._from_relative(x0_log + velocity_log, last_obs).permute(1, 0, 2)),
                to_degrees(gt_traj_abs.permute(1, 0, 2)),
            )
            .mean()
            .item()
        )

    return build_loss_report(
        model,
        total,
        l_cfm,
        l_reg,
        l_heading,
        l_calib,
        l_score,
        l_hard_reg,
        hard_score_mixture,
        precisions,
        ramp_reg,
        ramp_heading,
        ramp_calib,
        sigma,
        one_step_ade_km,
        difficulty_score,
        x0,
    )


def get_loss(model, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
    return get_loss_breakdown(model, batch_list, epoch=epoch)["total"]

_unwrap = unwrap_compiled
_norm_to_deg = to_degrees
_haversine_deg = haversine_km
_forward_azimuth = forward_bearing
_step_speeds_kmh = step_speeds_kmh
hard_score_from_obs = difficulty_score_from_history
_physics_score = physics_plausibility_score
compute_obs_attribution = compute_context_attribution
compute_cte_contribution = compute_ate_cte_decomposition
heading_loss_ms = heading_consistency_loss
reg_loss = displacement_regression_loss
compute_heading_loss = heading_loss_with_anchor
compute_calib_loss = speed_calibration_loss
compute_score_loss = candidate_selection_loss
kendall_combine = combine_losses_kendall
build_loss_dict = build_loss_report
