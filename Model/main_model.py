from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Main_model.ema import EMAModel
from Model.Main_model.velocity_transformer import VelocityTransformer
from Model.Main_model.context_encoder import ContextEncoder
from Model.Main_model.loss import (
    to_degrees,
    haversine_km,
    step_speeds_kmh,
    difficulty_score_from_history,
    physics_plausibility_score,
    compute_ensemble_uncertainty,
    compute_heading_deviation,
    compute_ate_cte_decomposition,
    get_loss_breakdown as _loss_get_loss_breakdown,
    get_loss as _loss_get_loss,
)


N_TOP_CANDIDATES_BLENDED = 3


class Main_model(nn.Module):

    def __init__(
        self,
        pred_len: int = 12,
        obs_len: int = 8,
        unet_in_ch: int = 13,
        d_cond: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_dec_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        sigma_min: float = 0.06,
        sigma_max: float = 0.15,
        sigma_decay_start: int = 5,
        sigma_decay_end: int = 100,
        lambda_reg: float = 0.2,
        lambda_heading: float = 0.07,
        lambda_momentum: float = 0.0,
        lambda_calib: float = 0.1,
        lambda_hard_reg: float = 0.02,
        log_sigma_reg_min_clamp: float = -3.0,
        enable_horizon_nll: bool = True,
        use_ot: bool = True,
        ot_epsilon: float = 0.05,
        use_ema: bool = True,
        ema_decay: float = 0.995,
        n_inference_steps: int = 10,
        n_ensemble: int = 20,
        sigma_inference: float = 0.04,
        use_curvature_score_train: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.use_curvature_score_train = use_curvature_score_train
        self.pred_len = pred_len
        self.obs_len = obs_len
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_decay_start = sigma_decay_start
        self.sigma_decay_end = sigma_decay_end
        self.lambda_reg = lambda_reg
        self.lambda_heading = lambda_heading
        self.lambda_momentum = 0.0
        self.lambda_calib = lambda_calib
        self.lambda_hard_reg = lambda_hard_reg
        self.log_sigma_reg_min_clamp = log_sigma_reg_min_clamp
        self.enable_horizon_nll = enable_horizon_nll
        self.use_ot = use_ot
        self.ot_epsilon = ot_epsilon
        self.n_inference_steps = n_inference_steps
        self.n_ensemble = n_ensemble
        self.sigma_inference = sigma_inference

        self.encoder = ContextEncoder(obs_len=obs_len, era5_in_channels=unet_in_ch, d_cond=d_cond)
        self.velocity = VelocityTransformer(
            pred_len=pred_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_dec_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            d_cond=d_cond,
        )
        self.use_ema = use_ema
        self._ema = None

        self.speed_correction_logits = nn.Parameter(torch.zeros(pred_len))
        self.obs_speed_calib_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, pred_len)
        )
        nn.init.zeros_(self.obs_speed_calib_mlp[-1].weight)
        nn.init.zeros_(self.obs_speed_calib_mlp[-1].bias)

       
        self.register_buffer("reg_dist_ema", torch.full((pred_len,), 100.0))
        self.reg_dist_ema_decay = 0.98
        self.register_buffer("reg_dist_ema_warmed", torch.zeros(1))

        self.register_buffer("heading_err_ema", torch.full((pred_len,), 1.0))
        self.heading_err_ema_decay = 0.98
        self.register_buffer("heading_err_ema_warmed", torch.zeros(1))

      
        self.hard_score_weight_logits = nn.Parameter(
            torch.log(torch.tensor([0.40, 0.30, 0.30, 0.15]))
        )

    
        self.score_weight_logits = nn.Parameter(
            torch.log(torch.tensor([0.30, 0.25, 0.30, 0.15, 0.01]))
        )
        self.score_v_sigma_scale_logit = nn.Parameter(torch.zeros(1))
        self.score_kernel_scale_logits = nn.Parameter(
            torch.log(torch.exp(torch.tensor([5.0, 3.0, 1.5])) - 1.0)
        )
        self.disp_decel_logit = nn.Parameter(torch.tensor(1.0986122886681098))

   
        self.log_sigma_reg = nn.Parameter(torch.tensor(-0.5 * math.log(2.0 * 0.20)))
        self.log_sigma_heading = nn.Parameter(torch.tensor(-0.5 * math.log(2.0 * 0.07)))
        self.log_sigma_calib = nn.Parameter(torch.tensor(-0.5 * math.log(2.0 * 0.10)))
        self.log_sigma_score = nn.Parameter(torch.tensor(-0.5 * math.log(2.0 * 0.10)))

    def init_ema(self):
        if self.use_ema:
            self._ema = EMAModel(self, decay=0.995)

    def ema_update(self):
        if self._ema is not None:
            self._ema.update(self)

    def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
        return x_abs - last_obs.unsqueeze(1)

    def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
        return x_rel + last_obs.unsqueeze(1)

    def _sigma_schedule(self, epoch: int) -> float:
        if epoch < self.sigma_decay_start:
            return self.sigma_max
        if epoch < self.sigma_decay_end:
            span = max(self.sigma_decay_end - self.sigma_decay_start, 1)
            progress = (epoch - self.sigma_decay_start) / span
            return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (
                1 + math.cos(math.pi * progress)
            )
        return self.sigma_min

    def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
        return _loss_get_loss_breakdown(self, batch_list, epoch=epoch, **kwargs)

    def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
        return _loss_get_loss(self, batch_list, epoch=epoch, **kwargs)

    def speed_calibrate_pred(
        self, pred_abs_norm: torch.Tensor, last_obs_norm: torch.Tensor, obs_norm: torch.Tensor
    ) -> torch.Tensor:
        if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
            return pred_abs_norm

        n_pred_steps, batch_size = pred_abs_norm.shape[0], pred_abs_norm.shape[1]
        obs_deg = to_degrees(obs_norm[..., :2])
        mean_obs_speed = step_speeds_kmh(obs_deg).mean(0)
        mean_obs_speed = torch.nan_to_num(mean_obs_speed, nan=5.0, posinf=100.0, neginf=5.0)
        mean_obs_speed_norm = (mean_obs_speed / 20.0).clamp(0.0, 3.0).view(batch_size, 1)

        horizon_offset = self.speed_correction_logits[:n_pred_steps].view(1, n_pred_steps)
        storm_correction = self.obs_speed_calib_mlp(mean_obs_speed_norm.to(pred_abs_norm.dtype))[
            :, :n_pred_steps
        ]
        correction_factor = (
            (torch.sigmoid(horizon_offset + storm_correction) * 2.0)
            .to(pred_abs_norm.dtype)
            .permute(1, 0)
            .unsqueeze(-1)
        )

        points = torch.cat([last_obs_norm.unsqueeze(0), pred_abs_norm], 0)
        raw_displacement = points[1:] - points[:-1]
        calibrated_displacement = raw_displacement * correction_factor

        calibrated = torch.empty_like(pred_abs_norm)
        running_position = last_obs_norm
        for h in range(n_pred_steps):
            running_position = running_position + calibrated_displacement[h]
            calibrated[h] = running_position
        return calibrated

    @staticmethod
    def _blend_top_k_candidates(
        all_candidates: torch.Tensor, scores: torch.Tensor, top_k: int
    ) -> torch.Tensor:
      
        SCORE_SOFTMAX_SHARPNESS = 3.0  

        batch_size = all_candidates.shape[2]
        top_indices = scores.topk(top_k, dim=0).indices
        blended = torch.zeros_like(all_candidates[0])
        for b in range(batch_size):
            idx_b = top_indices[:, b]
            weights_b = F.softmax(scores[idx_b, b] * SCORE_SOFTMAX_SHARPNESS, dim=0)
            blended[:, b, :] = (all_candidates[idx_b, :, b, :] * weights_b.view(top_k, 1, 1)).sum(0)
        return blended

    @torch.no_grad()
    def sample(
        self,
        batch_list,
        num_ensemble: Optional[int] = None,
        ddim_steps: Optional[int] = None,
        return_xai: bool = False,
        use_speed_calibration: bool = True,
        use_curvature_score: bool = False,
        return_attn: bool = False,
        **kwargs,
    ) -> Tuple:
       
        n_candidates = num_ensemble or self.n_ensemble
        n_euler_steps = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
        step_size = 1.0 / max(n_euler_steps, 1)

        obs_traj = batch_list[0]
        _, batch_size, _ = obs_traj.shape
        device = obs_traj.device

        difficulty_score = difficulty_score_from_history(
            obs_traj[:, :, :2], weight_logits=self.hard_score_weight_logits
        )
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0 = torch.zeros(batch_size, device=device)
        context = self.encoder(batch_list, hard_score=difficulty_score)

        all_candidates = self._generate_candidates(
            context, last_obs, batch_size, n_candidates, n_euler_steps, step_size, device, t0
        )

        scores = torch.stack(
            [
                physics_plausibility_score(
                    candidate,
                    obs_norm,
                    use_turning_rate_criterion=use_curvature_score,
                    weight_logits=self.score_weight_logits,
                    speed_sigma_scale_logit=self.score_v_sigma_scale_logit,
                    kernel_scale_logits=self.score_kernel_scale_logits,
                    displacement_decel_logit=self.disp_decel_logit,
                )
                for candidate in all_candidates
            ],
            0,
        )
        stacked_candidates = torch.stack(all_candidates, 0)
        top_k = min(N_TOP_CANDIDATES_BLENDED, n_candidates)
        pred_mean = self._blend_top_k_candidates(stacked_candidates, scores, top_k)

        if use_speed_calibration:
            pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

        if not return_xai:
            return pred_mean, torch.zeros_like(pred_mean), stacked_candidates

        xai = self._compute_xai(
            batch_list,
            context,
            obs_traj,
            obs_norm,
            last_obs,
            pred_mean,
            stacked_candidates,
            difficulty_score,
            return_attn,
            device,
            batch_size,
        )
        return pred_mean, torch.zeros_like(pred_mean), stacked_candidates, xai

    def _generate_candidates(
        self, context, last_obs, batch_size, n_candidates, n_euler_steps, step_size, device, t0
    ) -> List[torch.Tensor]:
 
        all_candidates = []
        for _ in range(n_candidates):
            z = torch.randn(batch_size, self.pred_len, 2, device=device) * self.sigma_inference
            if n_euler_steps <= 1:
                velocity = self.velocity(z, t0, context)
                z = z + velocity
            else:
                for step in range(n_euler_steps):
                    tau = torch.full((batch_size,), step * step_size, device=device)
                    z = (z + step_size * self.velocity(z, tau, context)).clamp(-3.0, 3.0)
            z = z.clamp(-20.0, 20.0)
            all_candidates.append(self._from_relative(z, last_obs).permute(1, 0, 2))
        return all_candidates

    def _compute_xai(
        self,
        batch_list,
        context,
        obs_traj,
        obs_norm,
        last_obs,
        pred_mean,
        all_candidates,
        difficulty_score,
        return_attn,
        device,
        batch_size,
    ):
     
        xai = {}
        xai.update(compute_ensemble_uncertainty(all_candidates))

        _, hard_score_components = difficulty_score_from_history(
            obs_norm, return_components=True, weight_logits=self.hard_score_weight_logits
        )
        xai["hard_components"] = hard_score_components

        if return_attn:
            zero_query = torch.zeros(batch_size, self.pred_len, 2, device=device)
            _, cross_attn = self.velocity(
                zero_query, torch.zeros(batch_size, device=device), context, return_attn=True
            )
            xai["cross_attn"] = cross_attn
            gamma_weight = self.velocity.film_gamma.weight.detach()
            beta_weight = self.velocity.film_beta.weight.detach()
           
            xai["film_gamma_deviation_per_horizon"] = (gamma_weight - 1.0).norm(dim=-1).tolist()
            xai["film_beta_deviation_per_horizon"] = beta_weight.norm(dim=-1).tolist()

        pred_deg = to_degrees(pred_mean)
        obs_deg = to_degrees(obs_norm)
        obs_speed = step_speeds_kmh(obs_deg)
        mean_obs_speed = obs_speed.mean(0)

        if pred_deg.shape[0] >= 2:
            joined_points = torch.cat([obs_deg[-1].unsqueeze(0), pred_deg], 0)
            mean_pred_speed = step_speeds_kmh(joined_points).mean(0)
        else:
            mean_pred_speed = mean_obs_speed.clone()

        speed_ratio = mean_pred_speed / mean_obs_speed.clamp(min=1.0)
        xai["speed_comparison"] = {
            "obs_speed_mean": float(mean_obs_speed.mean()),
            "pred_speed_mean": float(mean_pred_speed.mean()),
            "speed_ratio": float(speed_ratio.mean()),
            "per_storm_obs": mean_obs_speed,
            "per_storm_pred": mean_pred_speed,
            "over_predict": speed_ratio > 1.15,
            "under_predict": speed_ratio < 0.85,
        }

        xai["physics_components"] = self._compute_physics_components(
            pred_deg, pred_mean, obs_norm, mean_obs_speed, mean_pred_speed, batch_size, device
        )

        gt_traj = batch_list[1]
        gt_deg = to_degrees(gt_traj[:, :, :2])
        xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg)
        xai["ate_cte_decomp"] = compute_ate_cte_decomposition(pred_deg, gt_deg)
        xai["speed_per_horizon"] = self._compute_speed_per_horizon(pred_deg, gt_deg, obs_deg)
        xai["storm_categories"] = self._compute_storm_categories(obs_deg, pred_deg, gt_deg)
        xai["learned_params"] = self._compute_learned_params()
        return xai

    def _compute_physics_components(
        self, pred_deg, pred_mean, obs_norm, mean_obs_speed, mean_pred_speed, batch_size, device
    ):
        reference_speed = mean_obs_speed.clamp(min=5.0)
        speed_sigma = reference_speed * 0.5
        speed_score = torch.exp(-((mean_pred_speed - reference_speed) / speed_sigma).pow(2) * 0.5)

        if pred_deg.shape[0] >= 3:
            velocity = pred_deg[1:] - pred_deg[:-1]
            acceleration = (velocity[1:] - velocity[:-1]).norm(dim=-1)
            smoothness_score = torch.exp(-acceleration.mean(0) * 5.0)
        else:
            smoothness_score = torch.ones(batch_size, device=device)

        if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
            obs_velocity = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
            pred_velocity = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
            cosine_similarity = (
                F.normalize(obs_velocity, dim=-1, eps=1e-6) * F.normalize(pred_velocity, dim=-1, eps=1e-6)
            ).sum(-1)
            heading_score = torch.exp((cosine_similarity.clamp(-1, 1) - 1.0) * 3.0)
        else:
            heading_score = torch.ones(batch_size, device=device)

        return {
            "speed": speed_score,
            "smooth": smoothness_score,
            "heading": heading_score,
            "obs_speed": mean_obs_speed,
            "pred_speed": mean_pred_speed,
        }

    @staticmethod
    def _compute_speed_per_horizon(pred_deg, gt_deg, obs_deg):
        if not (pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2):
            return {}

        n_steps = min(pred_deg.shape[0], gt_deg.shape[0])
        last_obs_point = obs_deg[-1]
        pred_points = torch.cat([last_obs_point.unsqueeze(0), pred_deg[:n_steps]], 0)
        gt_points = torch.cat([last_obs_point.unsqueeze(0), gt_deg[:n_steps]], 0)
        pred_speed = step_speeds_kmh(pred_points)
        gt_speed = step_speeds_kmh(gt_points)
        ratio = pred_speed / gt_speed.clamp(min=1.0)
        mean_ratio_per_horizon = ratio.mean(1)

        def _horizon_mean(series, lo, hi):
            hi = min(hi, series.shape[0])
            return float(series[lo:hi].mean()) if hi > lo else float("nan")

        return {
            "ratio": mean_ratio_per_horizon.tolist(),
            "pred_kmh": pred_speed.mean(1).tolist(),
            "gt_kmh": gt_speed.mean(1).tolist(),
            "12h_ratio": _horizon_mean(mean_ratio_per_horizon, 0, 2),
            "24h_ratio": _horizon_mean(mean_ratio_per_horizon, 2, 4),
            "48h_ratio": _horizon_mean(mean_ratio_per_horizon, 6, 8),
            "72h_ratio": _horizon_mean(mean_ratio_per_horizon, 10, 12),
        }

    @staticmethod
    def _compute_storm_categories(obs_deg, pred_deg, gt_deg):
        
        obs_speed = step_speeds_kmh(obs_deg).mean(0)
        is_slow = obs_speed < 8.0
        is_medium = (obs_speed >= 8.0) & (obs_speed < 15.0)
        is_fast = obs_speed >= 15.0
        error_per_storm = haversine_km(
            pred_deg[: gt_deg.shape[0]], gt_deg[: pred_deg.shape[0]]
        ).mean(0)

        def _mean_error(mask):
            return float(error_per_storm[mask].mean()) if mask.sum() > 0 else float("nan")

        return {
            "n_slow": int(is_slow.sum()),
            "n_medium": int(is_medium.sum()),
            "n_fast": int(is_fast.sum()),
            "speed_mean": float(obs_speed.mean()),
            "speed_std": float(obs_speed.std()),
            "ade_slow": _mean_error(is_slow),
            "ade_medium": _mean_error(is_medium),
            "ade_fast": _mean_error(is_fast),
        }

    def _compute_learned_params(self):
        return {
            "speed_correction": (torch.sigmoid(self.speed_correction_logits) * 2.0).tolist(),
            "reg_dist_ema_km_per_horizon": self.reg_dist_ema.detach().tolist(),
            "heading_err_ema_per_horizon": self.heading_err_ema.detach().tolist(),
            "hard_score_weights": F.softmax(self.hard_score_weight_logits, dim=0).tolist(),
            "sigma_inf": float(self.sigma_inference),
            "log_sigma_reg": float(self.log_sigma_reg.detach()),
            "log_sigma_heading": float(self.log_sigma_heading.detach()),
            "log_sigma_calib": float(self.log_sigma_calib.detach()),
            "eff_lambda_reg": float(
                (
                    0.5
                    * torch.exp(-2.0 * self.log_sigma_reg.clamp(min=self.log_sigma_reg_min_clamp))
                ).detach()
            ),
            "eff_lambda_heading": float(
                (0.5 * torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))).detach()
            ),
            "eff_lambda_calib": float(
                (0.5 * torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))).detach()
            ),
        }

    @torch.no_grad()
    def sample_multiscale(
        self,
        batch_list,
        sigmas: Optional[List[float]] = None,
        n_per_sigma: int = 4,
        use_speed_calibration: bool = True,
        use_curvature_score: bool = False,
    ) -> Tuple:

        if sigmas is None:
            sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

        obs_traj = batch_list[0]
        batch_size = obs_traj.shape[1]
        device = obs_traj.device

        difficulty_score = difficulty_score_from_history(
            obs_traj[:, :, :2], weight_logits=self.hard_score_weight_logits
        )
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0 = torch.zeros(batch_size, device=device)
        context = self.encoder(batch_list, hard_score=difficulty_score)

        all_candidates = []
        for sigma in sigmas:
            for _ in range(n_per_sigma):
                z0 = torch.randn(batch_size, self.pred_len, 2, device=device) * sigma
                velocity = self.velocity(z0, t0, context)
                z_abs = (z0 + velocity).clamp(-20.0, 20.0)
                all_candidates.append(self._from_relative(z_abs, last_obs).permute(1, 0, 2))

        scores = torch.stack(
            [
                physics_plausibility_score(
                    candidate,
                    obs_norm,
                    use_turning_rate_criterion=use_curvature_score,
                    weight_logits=self.score_weight_logits,
                    speed_sigma_scale_logit=self.score_v_sigma_scale_logit,
                    kernel_scale_logits=self.score_kernel_scale_logits,
                    displacement_decel_logit=self.disp_decel_logit,
                )
                for candidate in all_candidates
            ],
            0,
        )
        stacked_candidates = torch.stack(all_candidates, 0)
        top_k = min(5, len(all_candidates))
        pred_mean = self._blend_top_k_candidates(stacked_candidates, scores, top_k)

        if use_speed_calibration:
            pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

        return pred_mean, torch.zeros_like(pred_mean), stacked_candidates


from Model.Main_model.augmentation import augment_batch

__all__ = ["Main_model", "EMAModel", "augment_batch"]
