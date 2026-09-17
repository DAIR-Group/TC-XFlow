
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:

    batch_size, seq_len, d_inner = u.shape
    d_state = A.shape[1]
    device = u.device

    delta_clamped = delta.clamp(-10.0, 1.0)

    discretized_A = torch.exp(torch.einsum("bti,is->btis", delta_clamped, A))
    discretized_Bu = torch.einsum("bti,bts->btis", delta_clamped, B) * u.unsqueeze(-1)

    hidden_state = torch.zeros(batch_size, d_inner, d_state, device=device)
    outputs = []
    for t in range(seq_len):
        hidden_state = discretized_A[:, t] * hidden_state + discretized_Bu[:, t]
        y_t = torch.einsum("bis,bs->bi", hidden_state, C[:, t])
        outputs.append(y_t)

    scan_output = torch.stack(outputs, dim=1)
    return scan_output + u * D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.depthwise_conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.state_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        dt_init_std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        seq_len = x.shape[1]

        gate_and_state = self.in_proj(x)
        x_state = gate_and_state[:, :, : self.d_inner]
        gate = gate_and_state[:, :, self.d_inner :]

        x_conv_out = self.depthwise_conv(x_state.permute(0, 2, 1))
        x_conv = F.silu(x_conv_out[:, :, :seq_len]).permute(0, 2, 1)

        dt_rank = self.state_proj.out_features - 2 * self.d_state
        state_params = self.state_proj(x_conv)
        dt_raw = state_params[:, :, :dt_rank]
        B_ssm = state_params[:, :, dt_rank : dt_rank + self.d_state]
        C_ssm = state_params[:, :, dt_rank + self.d_state :]

        delta = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log.float())

        y = selective_scan(x_conv, delta, A, B_ssm, C_ssm, self.D)
        y = y * F.silu(gate)
        y = self.out_proj(y)

        return self.drop(y) + residual


class MambaEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 128,
        d_model: int = 128,
        n_layers: int = 3,
        d_state: int = 16,
        dropout: float = 0.1,
        pool: str = "last",
    ):
        super().__init__()
        self.pool = pool

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        if self.pool == "last":
            pooled = h[:, -1, :]
        elif self.pool == "mean":
            pooled = h.mean(dim=1)
        else:
            pooled = h.max(dim=1).values
        return self.out_proj(pooled)


class BestTrackMambaEncoder(nn.Module):

    def __init__(
        self,
        bt_feature_dim: int = 4,
        era5_bottleneck_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 128,
        mamba_layers: int = 3,
        dropout: float = 0.1,
        d_state: int = 16,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.era5_bottleneck_dim = era5_bottleneck_dim

        self.bt_proj = nn.Sequential(
            nn.Linear(bt_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.fuse_bt_era5 = nn.Sequential(
            nn.Linear(era5_bottleneck_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
        )
        self.mamba = MambaEncoder(
            input_dim=hidden_dim * 2,
            hidden_dim=output_dim,
            d_model=output_dim,
            n_layers=mamba_layers,
            d_state=d_state,
            dropout=dropout,
            pool="last",
        )

    def forward(self, bt_and_meta_obs: torch.Tensor, era5_bottleneck: torch.Tensor) -> torch.Tensor:

        n_obs_steps = bt_and_meta_obs.shape[1]
        n_bottleneck_steps = era5_bottleneck.shape[1]

        if n_bottleneck_steps != n_obs_steps:
            era5_bottleneck = F.interpolate(
                era5_bottleneck.permute(0, 2, 1),
                size=n_obs_steps,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        bt_embedded = self.bt_proj(bt_and_meta_obs)
        fused = self.fuse_bt_era5(torch.cat([era5_bottleneck, bt_embedded], dim=-1))
        return self.mamba(fused)
