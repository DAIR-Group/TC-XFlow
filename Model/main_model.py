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
    _norm_to_deg,
    _haversine_deg,
    _step_speeds_kmh,
    hard_score_from_obs,
    _physics_score,
    compute_ensemble_uncertainty,
    compute_heading_deviation,
    compute_cte_contribution,
    get_loss_breakdown as _loss_get_loss_breakdown,
    get_loss as _loss_get_loss,
)


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

        self.encoder = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
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
            t = (epoch - self.sigma_decay_start) / span
            return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (
                1 + math.cos(math.pi * t)
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

        T, B = pred_abs_norm.shape[0], pred_abs_norm.shape[1]
        obs_deg_c = _norm_to_deg(obs_norm[..., :2])
        obs_spd_mu = _step_speeds_kmh(obs_deg_c).mean(0)
        obs_spd_mu = torch.nan_to_num(obs_spd_mu, nan=5.0, posinf=100.0, neginf=5.0)
        obs_spd_norm = (obs_spd_mu / 20.0).clamp(0.0, 3.0).view(B, 1)

        base_logit = self.speed_correction_logits[:T].view(1, T)
        delta_logit = self.obs_speed_calib_mlp(obs_spd_norm.to(pred_abs_norm.dtype))[:, :T]
        correction = (
            (torch.sigmoid(base_logit + delta_logit) * 2.0)
            .to(pred_abs_norm.dtype)
            .permute(1, 0)
            .unsqueeze(-1)
        )

        pts = torch.cat([last_obs_norm.unsqueeze(0), pred_abs_norm], 0)
        disp = pts[1:] - pts[:-1]
        disp_cal = disp * correction

        out = torch.empty_like(pred_abs_norm)
        cur = last_obs_norm
        for t in range(T):
            cur = cur + disp_cal[t]
            out[t] = cur
        return out

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

        K = num_ensemble or self.n_ensemble
        N = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
        dt = 1.0 / max(N, 1)

        obs_traj = batch_list[0]
        T_obs, B, _ = obs_traj.shape
        device = obs_traj.device

        h_score = hard_score_from_obs(
            obs_traj[:, :, :2], weight_logits=self.hard_score_weight_logits
        )
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0 = torch.zeros(B, device=device)
        cond = self.encoder(batch_list, hard_score=h_score)

        all_traj = self._generate_candidates(cond, last_obs, B, K, N, dt, device, t0)

        scores = torch.stack(
            [
                _physics_score(
                    t,
                    obs_norm,
                    use_curvature_score=use_curvature_score,
                    weight_logits=self.score_weight_logits,
                    v_sigma_scale_logit=self.score_v_sigma_scale_logit,
                    kernel_scale_logits=self.score_kernel_scale_logits,
                    disp_decel_logit=self.disp_decel_logit,
                )
                for t in all_traj
            ],
            0,
        )
        all_t = torch.stack(all_traj, 0)
        pred_mean = self._top3_weighted_average(all_t, scores, B, K)

        if use_speed_calibration:
            pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

        if not return_xai:
            return pred_mean, torch.zeros_like(pred_mean), all_t

        xai = self._compute_xai(
            batch_list,
            cond,
            obs_traj,
            obs_norm,
            last_obs,
            pred_mean,
            all_t,
            h_score,
            return_attn,
            device,
            B,
        )
        return pred_mean, torch.zeros_like(pred_mean), all_t, xai

    def _generate_candidates(self, cond, last_obs, B, K, N, dt, device, t0):

        all_traj = []
        for _ in range(K):
            x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference
            if N <= 1:
                v = self.velocity(x_rel, t0, cond)
                x_rel = x_rel + v
            else:
                for step in range(N):
                    t_b = torch.full((B,), step * dt, device=device)
                    x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3.0, 3.0)
            x_rel = x_rel.clamp(-20.0, 20.0)
            all_traj.append(self._from_relative(x_rel, last_obs).permute(1, 0, 2))
        return all_traj

    @staticmethod
    def _top3_weighted_average(all_t, scores, B, K):
        top_k = min(3, K)
        top_idx = scores.topk(top_k, dim=0).indices
        pred_mean = torch.zeros_like(all_t[0])
        for b in range(B):
            idx_b = top_idx[:, b]
            w_b = F.softmax(scores[idx_b, b] * 3.0, dim=0)
            pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)
        return pred_mean

    def _compute_xai(
        self,
        batch_list,
        cond,
        obs_traj,
        obs_norm,
        last_obs,
        pred_mean,
        all_t,
        h_score,
        return_attn,
        device,
        B,
    ):

        xai = {}
        xai.update(compute_ensemble_uncertainty(all_t))

        _, hard_comps = hard_score_from_obs(
            obs_norm, return_components=True, weight_logits=self.hard_score_weight_logits
        )
        xai["hard_components"] = hard_comps

        if return_attn:
            x0_attn = torch.zeros(B, self.pred_len, 2, device=device)
            _, attn_stack = self.velocity(
                x0_attn, torch.zeros(B, device=device), cond, return_attn=True
            )
            xai["cross_attn"] = attn_stack
            gamma_w = self.velocity.film_gamma.weight.detach()
            beta_w = self.velocity.film_beta.weight.detach()
            xai["film_gamma_deviation_per_horizon"] = (gamma_w - 1.0).norm(dim=-1).tolist()
            xai["film_beta_deviation_per_horizon"] = beta_w.norm(dim=-1).tolist()

        pred_deg = _norm_to_deg(pred_mean)
        obs_deg_x = _norm_to_deg(obs_norm)
        obs_spd_x = _step_speeds_kmh(obs_deg_x)
        obs_spd_mu = obs_spd_x.mean(0)

        if pred_deg.shape[0] >= 2:
            pts_x = torch.cat([obs_deg_x[-1].unsqueeze(0), pred_deg], 0)
            pred_spd_mu = _step_speeds_kmh(pts_x).mean(0)
        else:
            pred_spd_mu = obs_spd_mu.clone()

        speed_ratio = pred_spd_mu / obs_spd_mu.clamp(min=1.0)
        xai["speed_comparison"] = {
            "obs_speed_mean": float(obs_spd_mu.mean()),
            "pred_speed_mean": float(pred_spd_mu.mean()),
            "speed_ratio": float(speed_ratio.mean()),
            "per_storm_obs": obs_spd_mu,
            "per_storm_pred": pred_spd_mu,
            "over_predict": speed_ratio > 1.15,
            "under_predict": speed_ratio < 0.85,
        }

        xai["physics_components"] = self._compute_physics_components(
            pred_deg, pred_mean, obs_norm, obs_spd_mu, pred_spd_mu, B, device
        )

        gt_traj_xai = batch_list[1]
        gt_deg_xai = _norm_to_deg(gt_traj_xai[:, :, :2])
        xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)
        xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)
        xai["speed_per_horizon"] = self._compute_speed_per_horizon(pred_deg, gt_deg_xai, obs_deg_x)
        xai["storm_categories"] = self._compute_storm_categories(obs_deg_x, pred_deg, gt_deg_xai)
        xai["learned_params"] = self._compute_learned_params()
        return xai

    def _compute_physics_components(
        self, pred_deg, pred_mean, obs_norm, obs_spd_mu, pred_spd_mu, B, device
    ):
        v_ref = obs_spd_mu.clamp(min=5.0)
        v_sig = v_ref * 0.5
        spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)

        if pred_deg.shape[0] >= 3:
            vel_x = pred_deg[1:] - pred_deg[:-1]
            accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
            smo_sc = torch.exp(-accel_x.mean(0) * 5.0)
        else:
            smo_sc = torch.ones(B, device=device)

        if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
            ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
            pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
            cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
            hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
        else:
            hd_sc = torch.ones(B, device=device)

        return {
            "speed": spd_sc,
            "smooth": smo_sc,
            "heading": hd_sc,
            "obs_speed": obs_spd_mu,
            "pred_speed": pred_spd_mu,
        }

    @staticmethod
    def _compute_speed_per_horizon(pred_deg, gt_deg_xai, obs_deg_x):
        if not (pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2):
            return {}

        T8 = min(pred_deg.shape[0], gt_deg_xai.shape[0])
        last_d = obs_deg_x[-1]
        pts_pred = torch.cat([last_d.unsqueeze(0), pred_deg[:T8]], 0)
        pts_gt = torch.cat([last_d.unsqueeze(0), gt_deg_xai[:T8]], 0)
        spd_pred = _step_speeds_kmh(pts_pred)
        spd_gt = _step_speeds_kmh(pts_gt)
        ratio = spd_pred / spd_gt.clamp(min=1.0)
        r_mean = ratio.mean(1)

        def _hz(s, lo, hi):
            hi = min(hi, s.shape[0])
            return float(s[lo:hi].mean()) if hi > lo else float("nan")

        return {
            "ratio": r_mean.tolist(),
            "pred_kmh": spd_pred.mean(1).tolist(),
            "gt_kmh": spd_gt.mean(1).tolist(),
            "12h_ratio": _hz(r_mean, 0, 2),
            "24h_ratio": _hz(r_mean, 2, 4),
            "48h_ratio": _hz(r_mean, 6, 8),
            "72h_ratio": _hz(r_mean, 10, 12),
        }

    @staticmethod
    def _compute_storm_categories(obs_deg_x, pred_deg, gt_deg_xai):
        obs_spd_cat = _step_speeds_kmh(obs_deg_x).mean(0)
        sm = obs_spd_cat < 8.0
        mm = (obs_spd_cat >= 8.0) & (obs_spd_cat < 15.0)
        fm = obs_spd_cat >= 15.0
        ade_per_storm = _haversine_deg(
            pred_deg[: gt_deg_xai.shape[0]], gt_deg_xai[: pred_deg.shape[0]]
        ).mean(0)

        def _cat_ade(mask):
            return float(ade_per_storm[mask].mean()) if mask.sum() > 0 else float("nan")

        return {
            "n_slow": int(sm.sum()),
            "n_medium": int(mm.sum()),
            "n_fast": int(fm.sum()),
            "speed_mean": float(obs_spd_cat.mean()),
            "speed_std": float(obs_spd_cat.std()),
            "ade_slow": _cat_ade(sm),
            "ade_medium": _cat_ade(mm),
            "ade_fast": _cat_ade(fm),
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
        B = obs_traj.shape[1]
        device = obs_traj.device

        h_score = hard_score_from_obs(
            obs_traj[:, :, :2], weight_logits=self.hard_score_weight_logits
        )
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0 = torch.zeros(B, device=device)
        cond = self.encoder(batch_list, hard_score=h_score)

        all_traj = []
        for sigma in sigmas:
            for _ in range(n_per_sigma):
                x0 = torch.randn(B, self.pred_len, 2, device=device) * sigma
                v = self.velocity(x0, t0, cond)
                x_abs = (x0 + v).clamp(-20.0, 20.0)
                all_traj.append(self._from_relative(x_abs, last_obs).permute(1, 0, 2))

        scores = torch.stack(
            [
                _physics_score(
                    t,
                    obs_norm,
                    use_curvature_score=use_curvature_score,
                    weight_logits=self.score_weight_logits,
                    v_sigma_scale_logit=self.score_v_sigma_scale_logit,
                    kernel_scale_logits=self.score_kernel_scale_logits,
                    disp_decel_logit=self.disp_decel_logit,
                )
                for t in all_traj
            ],
            0,
        )
        all_t = torch.stack(all_traj, 0)
        top_k = min(5, len(all_traj))
        top_idx = scores.topk(top_k, dim=0).indices

        pred_mean = torch.zeros_like(all_traj[0])
        for b in range(B):
            idx_b = top_idx[:, b]
            w_b = F.softmax(scores[idx_b, b] * 3.0, dim=0)
            pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

        if use_speed_calibration:
            pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

        return pred_mean, torch.zeros_like(pred_mean), all_t


from Model.Main_model.augmentation import augment_batch

__all__ = ["Main_model", "EMAModel", "augment_batch"]
