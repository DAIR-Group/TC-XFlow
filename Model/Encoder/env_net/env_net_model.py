from __future__ import annotations

import torch
import torch.nn as nn

from Model.Encoder.env_net.feature_encoding import ENV_1D_DIM, ENV_3D_DIM
from Model.Encoder.env_net.tensor_utils import build_env_vector


class Env_net(nn.Module):

    def __init__(self, obs_len: int = 8, embed_dim: int = 16, d_model: int = 64):
        super().__init__()
        self.obs_len = obs_len
        self.d_model = d_model
        H1, H2, H3 = 64, 32, 64

        self.mlp_env_1d = nn.Sequential(
            nn.Linear(ENV_1D_DIM, H1),
            nn.LayerNorm(H1),
            nn.GELU(),
            nn.Linear(H1, H1),
            nn.LayerNorm(H1),
            nn.GELU(),
        )
        self.cnn_env_3d = nn.Sequential(
            nn.Conv1d(ENV_3D_DIM, H2, 3, padding=1),
            nn.BatchNorm1d(H2),
            nn.GELU(),
            nn.Conv1d(H2, H2, 3, padding=1),
            nn.BatchNorm1d(H2),
            nn.GELU(),
        )
        self.mlp_fusion = nn.Sequential(nn.Linear(H1 + H2, H3), nn.LayerNorm(H3), nn.GELU())
        self.pos_enc_env = nn.Parameter(torch.randn(1, obs_len, H3) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=H3,
            nhead=4,
            dim_feedforward=H3 * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_env = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.out_proj = nn.Sequential(nn.Linear(H3, d_model), nn.LayerNorm(d_model))

        self._attn_cache: list = []
        self._xai_hooks_registered = False

    def _register_xai_hooks(self):

        if self._xai_hooks_registered:
            return

        def _pre_hook(module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            return args, kwargs

        def _make_hook(layer_idx: int):
            def _hook(module, inputs, output):
                if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                    self._attn_cache.append(
                        {
                            "layer": layer_idx,
                            "kind": "env_self_attn",
                            "weights": output[1].detach(),
                        }
                    )

            return _hook

        for i, layer in enumerate(self.transformer_env.layers):
            layer.self_attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            layer.self_attn.register_forward_hook(_make_hook(i))
        self._xai_hooks_registered = True

    def forward(self, env_data, gph: torch.Tensor, return_attention: bool = False):
        if return_attention:
            self._register_xai_hooks()
            self._attn_cache = []

        if gph.dim() == 4:
            gph = gph.unsqueeze(1)
        B, C, T, H, W = gph.shape
        device = gph.device

        feat = build_env_vector(env_data, B, T, device)
        feat_1d = feat[:, :, :ENV_1D_DIM]
        feat_3d = feat[:, :, ENV_1D_DIM:]

        e_1d = self.mlp_env_1d(feat_1d)
        e_3d = self.cnn_env_3d(feat_3d.permute(0, 2, 1)).permute(0, 2, 1)
        e_env = self.mlp_fusion(torch.cat([e_1d, e_3d], dim=-1))

        t_actual = min(T, self.pos_enc_env.shape[1])
        e_env = e_env[:, :t_actual, :] + self.pos_enc_env[:, :t_actual, :]
        e_env_time = self.transformer_env(e_env)
        ctx = self.out_proj(e_env_time[:, -1, :])

        if return_attention:
            return ctx, 0, 0, list(self._attn_cache)
        return ctx, 0, 0
