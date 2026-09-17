
from __future__ import annotations

import torch
import torch.nn as nn

from Model.Encoder.env_net.feature_encoding import ENV_DIRECT_DIM, ENV_ERA5_DERIVED_DIM
from Model.Encoder.env_net.tensor_utils import build_env_vector


class Env_net(nn.Module):
 

    def __init__(self, obs_len: int = 8, embed_dim: int = 16, d_model: int = 64):
        super().__init__()
        self.obs_len = obs_len
        self.d_model = d_model
        direct_hidden, era5_derived_hidden, fused_hidden = 64, 32, 64

        self.direct_feature_encoder = nn.Sequential(
            nn.Linear(ENV_DIRECT_DIM, direct_hidden),
            nn.LayerNorm(direct_hidden),
            nn.GELU(),
            nn.Linear(direct_hidden, direct_hidden),
            nn.LayerNorm(direct_hidden),
            nn.GELU(),
        )
     
        self.era5_derived_feature_encoder = nn.Sequential(
            nn.Conv1d(ENV_ERA5_DERIVED_DIM, era5_derived_hidden, 3, padding=1),
            nn.BatchNorm1d(era5_derived_hidden),
            nn.GELU(),
            nn.Conv1d(era5_derived_hidden, era5_derived_hidden, 3, padding=1),
            nn.BatchNorm1d(era5_derived_hidden),
            nn.GELU(),
        )
        self.fuse_descriptors = nn.Sequential(
            nn.Linear(direct_hidden + era5_derived_hidden, fused_hidden),
            nn.LayerNorm(fused_hidden),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.randn(1, obs_len, fused_hidden) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_hidden,
            nhead=4,
            dim_feedforward=fused_hidden * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Sequential(nn.Linear(fused_hidden, d_model), nn.LayerNorm(d_model))

        self._attention_cache: list = []
        self._xai_hooks_registered = False

    def _register_xai_hooks(self):
        if self._xai_hooks_registered:
            return

        def _request_attention_weights(module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            return args, kwargs

        def _make_capture_hook(layer_idx: int):
            def _capture_hook(module, inputs, output):
                if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                    self._attention_cache.append(
                        {
                            "layer": layer_idx,
                            "kind": "env_self_attn",
                            "weights": output[1].detach(),
                        }
                    )

            return _capture_hook

        for i, layer in enumerate(self.temporal_encoder.layers):
            layer.self_attn.register_forward_pre_hook(_request_attention_weights, with_kwargs=True)
            layer.self_attn.register_forward_hook(_make_capture_hook(i))
        self._xai_hooks_registered = True

    def forward(self, env_data, era5_patch: torch.Tensor, return_attention: bool = False):

        if return_attention:
            self._register_xai_hooks()
            self._attention_cache = []

        if era5_patch.dim() == 4:
            era5_patch = era5_patch.unsqueeze(1)
        batch_size, _, n_obs_steps, _, _ = era5_patch.shape
        device = era5_patch.device

        env_vector = build_env_vector(env_data, batch_size, n_obs_steps, device)
        direct_features = env_vector[:, :, :ENV_DIRECT_DIM]
        era5_derived_features = env_vector[:, :, ENV_DIRECT_DIM:]

        direct_encoded = self.direct_feature_encoder(direct_features)
        era5_derived_encoded = self.era5_derived_feature_encoder(
            era5_derived_features.permute(0, 2, 1)
        ).permute(0, 2, 1)
        fused = self.fuse_descriptors(torch.cat([direct_encoded, era5_derived_encoded], dim=-1))

        n_actual_steps = min(n_obs_steps, self.position_embedding.shape[1])
        fused = fused[:, :n_actual_steps, :] + self.position_embedding[:, :n_actual_steps, :]
        temporal_encoded = self.temporal_encoder(fused)
        context = self.output_proj(temporal_encoded[:, -1, :])

        if return_attention:
            return context, 0, 0, list(self._attention_cache)
        return context, 0, 0
