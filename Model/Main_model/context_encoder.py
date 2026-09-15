from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Encoder.FNO3D_encoder import FNO3DEncoder
from Model.Encoder.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.Encoder.env_net import Env_net
from Model.Main_model.loss import _norm_to_deg, _step_speeds_kmh


class ContextEncoder(nn.Module):
    RAW_CTX_DIM = 512

    def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
        super().__init__()
        self.obs_len = obs_len
        self.d_cond = d_cond

        self.spatial_enc = FNO3DEncoder(
            in_channel=unet_in_ch,
            out_channel=1,
            d_model=32,
            n_layers=4,
            modes_t=4,
            modes_h=4,
            modes_w=4,
            spatial_down=32,
            dropout=0.05,
        )
        self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.bottleneck_proj = nn.Linear(128, 128)
        self.decoder_proj = nn.Linear(1, 16)
        self.enc_1d = DataEncoder1D(
            in_1d=4,
            feat_3d_dim=128,
            mlp_h=64,
            lstm_hidden=128,
            lstm_layers=3,
            dropout=0.1,
            d_state=16,
        )
        self.env_enc = Env_net(obs_len=obs_len, d_model=32)
        self.ctx_fc1 = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
        self.ctx_ln = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.1)
        self.ctx_fc2 = nn.Linear(self.RAW_CTX_DIM, d_cond)
        self.ctx_ln2 = nn.LayerNorm(d_cond)

        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len * 7, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, d_cond // 2),
            nn.GELU(),
        )
        self.hard_embed = nn.Sequential(
            nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4)
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond), nn.LayerNorm(d_cond), nn.GELU()
        )

    def _encode_raw(self, batch_list) -> torch.Tensor:

        obs_traj = batch_list[0]
        obs_Me = batch_list[7]
        image_obs = batch_list[11]
        env_data = batch_list[13]
        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(2)
        if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
            image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]
        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1] != T_obs:
            e_3d_s = F.interpolate(
                e_3d_s.permute(0, 2, 1), size=T_obs, mode="linear", align_corners=False
            ).permute(0, 2, 1)

        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_w = torch.softmax(
            torch.arange(e_3d_dec_t.shape[1], dtype=torch.float, device=e_3d_dec_t.device) * 0.5,
            dim=0,
        )
        f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t = self.enc_1d(obs_in, e_3d_s)
        e_env, _, _ = self.env_enc(env_data, image_obs)
        return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

    def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
        B = obs_traj.shape[1]
        T_obs = obs_traj.shape[0]
        if T_obs >= 2:
            traj_deg = _norm_to_deg(obs_traj)
            vel_norm = obs_traj[1:] - obs_traj[:-1]
            speed = _step_speeds_kmh(traj_deg)
            speed_n = (speed / 20.0).clamp(-3.0, 3.0)
            heading = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
            if T_obs >= 3:
                dspd = speed[1:] - speed[:-1]
                accel = torch.cat([obs_traj.new_zeros(1, B), (dspd / 10.0).clamp(-3.0, 3.0)], 0)
                dh = torch.cat([obs_traj.new_zeros(1, B), heading[1:] - heading[:-1]], 0)
                turn_rate = torch.atan2(torch.sin(dh), torch.cos(dh)) / math.pi
            else:
                accel = obs_traj.new_zeros(T_obs - 1, B)
                turn_rate = obs_traj.new_zeros(T_obs - 1, B)
            kine = torch.stack(
                [
                    vel_norm[:, :, 0],
                    vel_norm[:, :, 1],
                    speed_n,
                    heading.sin(),
                    heading.cos(),
                    accel,
                    turn_rate,
                ],
                dim=-1,
            )
        else:
            kine = obs_traj.new_zeros(self.obs_len, B, 7)

        if kine.shape[0] < self.obs_len:
            kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 7), kine], 0)
        else:
            kine = kine[-self.obs_len :]
        return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

    def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        raw = self._encode_raw(batch_list)
        ctx = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
        kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
        if hard_score is None:
            hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
        hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
        return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))
