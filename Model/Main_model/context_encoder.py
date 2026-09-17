
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Encoder.FNO3D_encoder import FNO3DEncoder
from Model.Encoder.mamba_encoder import BestTrackMambaEncoder
from Model.Encoder.env_net import Env_net
from Model.Main_model.loss import _norm_to_deg, _step_speeds_kmh


@dataclass(frozen=True)
class BatchFields:
   
    bt_obs: Any
    bt_pred: Any
    bt_obs_rel: Any
    bt_pred_rel: Any
    non_linear_ped: Any
    mask: Any
    seq_start_end: Any
    bt_extra_obs: Any
    bt_extra_pred: Any
    bt_extra_obs_rel: Any
    bt_extra_pred_rel: Any
    era5_obs: Any
    era5_pred: Any
    env_features: Any
    reserved: Any
    storm_info: Any

    @classmethod
    def from_list(cls, batch_list: List[Any]) -> "BatchFields":
        return cls(*batch_list)


class ContextEncoder(nn.Module):
  
    FUSED_MODALITY_DIM = 512 

    def __init__(self, obs_len: int = 8, era5_in_channels: int = 13, d_cond: int = 256):
        super().__init__()
        self.obs_len = obs_len
        self.d_cond = d_cond

        self.era5_encoder = FNO3DEncoder(
            in_channel=era5_in_channels,
            out_channel=1,
            d_model=32,
            n_layers=4,
            modes_t=4,
            modes_h=4,
            modes_w=4,
            spatial_down=32,
            dropout=0.05,
        )
       
        self.era5_bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.era5_bottleneck_proj = nn.Linear(128, 128)
        self.era5_summary_proj = nn.Linear(1, 16)

       
        self.bt_era5_encoder = BestTrackMambaEncoder(
            bt_feature_dim=4,
            era5_bottleneck_dim=128,
            hidden_dim=64,
            output_dim=128,
            mamba_layers=3,
            dropout=0.1,
            d_state=16,
        )


        self.env_encoder = Env_net(obs_len=obs_len, d_model=32)


        self.fuse_modalities = nn.Linear(128 + 32 + 16, self.FUSED_MODALITY_DIM)
        self.fuse_modalities_norm = nn.LayerNorm(self.FUSED_MODALITY_DIM)
        self.fuse_modalities_drop = nn.Dropout(0.1)
        self.project_to_context = nn.Linear(self.FUSED_MODALITY_DIM, d_cond)
        self.project_to_context_norm = nn.LayerNorm(d_cond)

        self.kinematic_sequence_encoder = nn.Sequential(
            nn.Linear(obs_len * 7, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, d_cond // 2),
            nn.GELU(),
        )
       
        self.difficulty_score_encoder = nn.Sequential(
            nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4)
        )
        self.final_fuse = nn.Sequential(
            nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond), nn.LayerNorm(d_cond), nn.GELU()
        )

    def _encode_modalities(self, batch: BatchFields) -> torch.Tensor:


        era5_patch = batch.era5_obs
        if era5_patch.dim() == 4:
            era5_patch = era5_patch.unsqueeze(2)
        if era5_patch.shape[1] == 1 and self.era5_encoder.in_channel != 1:
            era5_patch = era5_patch.expand(-1, self.era5_encoder.in_channel, -1, -1, -1)

        era5_bottleneck_full, era5_decoder_summary = self.era5_encoder.encode(era5_patch)
        n_obs_steps = batch.bt_obs.shape[0]

        era5_bottleneck = (
            self.era5_bottleneck_pool(era5_bottleneck_full).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        )
        era5_bottleneck = self.era5_bottleneck_proj(era5_bottleneck)
        if era5_bottleneck.shape[1] != n_obs_steps:
            era5_bottleneck = F.interpolate(
                era5_bottleneck.permute(0, 2, 1),
                size=n_obs_steps,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)


        era5_decoder_seq = era5_decoder_summary.squeeze(1).squeeze(-1).squeeze(-1)
        recency_weights = torch.softmax(
            torch.arange(era5_decoder_seq.shape[1], dtype=torch.float, device=era5_decoder_seq.device)
            * 0.5,
            dim=0,
        )
        era5_summary = self.era5_summary_proj(
            (era5_decoder_seq * recency_weights.unsqueeze(0)).sum(1, keepdim=True)
        )

        bt_full = torch.cat([batch.bt_obs, batch.bt_extra_obs], dim=2).permute(1, 0, 2)
        h_n = self.bt_era5_encoder(bt_full, era5_bottleneck)

        # Environmental branch.
        e_env, _, _ = self.env_encoder(batch.env_features, era5_patch)

        c_global_raw = self.fuse_modalities(torch.cat([h_n, e_env, era5_summary], dim=-1))
        return F.gelu(self.fuse_modalities_norm(c_global_raw))

    def _encode_kinematic_sequence(self, bt_position_obs: torch.Tensor) -> torch.Tensor:
    
        batch_size = bt_position_obs.shape[1]
        n_obs_steps = bt_position_obs.shape[0]

        if n_obs_steps >= 2:
            position_deg = _norm_to_deg(bt_position_obs)
            step_displacement = bt_position_obs[1:] - bt_position_obs[:-1]
            speed = _step_speeds_kmh(position_deg)
            speed_norm = (speed / 20.0).clamp(-3.0, 3.0)
            heading = torch.atan2(step_displacement[:, :, 1], step_displacement[:, :, 0])

            if n_obs_steps >= 3:
                speed_delta = speed[1:] - speed[:-1]
                acceleration = torch.cat(
                    [bt_position_obs.new_zeros(1, batch_size), (speed_delta / 10.0).clamp(-3.0, 3.0)], 0
                )
                heading_delta = torch.cat(
                    [bt_position_obs.new_zeros(1, batch_size), heading[1:] - heading[:-1]], 0
                )
                turn_rate = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta)) / math.pi
            else:
                acceleration = bt_position_obs.new_zeros(n_obs_steps - 1, batch_size)
                turn_rate = bt_position_obs.new_zeros(n_obs_steps - 1, batch_size)

            kinematic_feature = torch.stack(
                [
                    step_displacement[:, :, 0],
                    step_displacement[:, :, 1],
                    speed_norm,
                    heading.sin(),
                    heading.cos(),
                    acceleration,
                    turn_rate,
                ],
                dim=-1,
            )
        else:
            kinematic_feature = bt_position_obs.new_zeros(self.obs_len, batch_size, 7)

        if kinematic_feature.shape[0] < self.obs_len:
            pad_len = self.obs_len - kinematic_feature.shape[0]
            kinematic_feature = torch.cat(
                [bt_position_obs.new_zeros(pad_len, batch_size, 7), kinematic_feature], 0
            )
        else:
            kinematic_feature = kinematic_feature[-self.obs_len :]

        return self.kinematic_sequence_encoder(
            kinematic_feature.permute(1, 0, 2).reshape(batch_size, -1)
        )

    def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:

        batch = BatchFields.from_list(batch_list)

        c_global = self._encode_modalities(batch)
        c_global = self.project_to_context_norm(
            self.project_to_context(self.fuse_modalities_drop(c_global))
        )

        kinematic_feature = self._encode_kinematic_sequence(batch.bt_obs[:, :, :2])

        if hard_score is None:
            hard_score = torch.zeros(c_global.shape[0], device=c_global.device)
        difficulty_embedding = self.difficulty_score_encoder(
            hard_score.unsqueeze(1).to(c_global.dtype)
        )

        return self.final_fuse(torch.cat([c_global, kinematic_feature, difficulty_embedding], dim=-1))
