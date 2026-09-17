from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityTransformer(nn.Module):
    def __init__(
        self,
        pred_len: int = 12,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        d_cond: int = 256,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.state_embed = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.horizon_embed = nn.Embedding(pred_len, d_model)
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.context_proj = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))

        
        self.film_gamma = nn.Embedding(pred_len, d_model)
        self.film_beta = nn.Embedding(pred_len, d_model)
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_beta.weight)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2)
        )
        self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def _flow_time_embedding(self, tau: torch.Tensor) -> torch.Tensor:
  
        half_dim = self.d_model // 2
        frequency = torch.exp(
            torch.arange(half_dim, device=tau.device, dtype=tau.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        angle = tau.float().unsqueeze(1) * frequency.unsqueeze(0)
        embedding = torch.cat([angle.sin(), angle.cos()], dim=-1)
        if self.d_model % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return self.flow_time_mlp(embedding)

    def _decode_with_cross_attention(self, query_embed: torch.Tensor, memory: torch.Tensor):
       
        x = query_embed
        attention_per_layer = []
        for layer in self.decoder.layers:
            self_attn_out = layer.self_attn(
                layer.norm1(x), layer.norm1(x), layer.norm1(x), need_weights=False
            )[0]
            x = x + layer.dropout1(self_attn_out)

            normed = layer.norm2(x)
            cross_attn_out, cross_attn_weights = layer.multihead_attn(
                normed, memory, memory, need_weights=True, average_attn_weights=True
            )
            attention_per_layer.append(cross_attn_weights)
            x = x + layer.dropout2(cross_attn_out)

            feedforward_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x)))))
            x = x + layer.dropout3(feedforward_out)

        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x, torch.stack(attention_per_layer, dim=0)

    def forward(
        self, z_tau: torch.Tensor, tau: torch.Tensor, context: torch.Tensor, return_attn: bool = False
    ):
       
        batch_size, n_horizons, _ = z_tau.shape
        horizon_idx = torch.arange(n_horizons, device=z_tau.device).unsqueeze(0).expand(batch_size, -1)
        query = self.state_embed(z_tau) + self.pos_embed[:, :n_horizons] + self.horizon_embed(horizon_idx)

        context_vec = self.context_proj(context)

        gamma = self.film_gamma(horizon_idx[0]).unsqueeze(0)
        beta = self.film_beta(horizon_idx[0]).unsqueeze(0)
        query = query + (gamma * context_vec.unsqueeze(1) + beta)  # Eq. film

        
        memory = torch.cat([self._flow_time_embedding(tau).unsqueeze(1), context_vec.unsqueeze(1)], dim=1)

        if return_attn:
            decoded, attention_stack = self._decode_with_cross_attention(query, memory)
            decoded = self.out_norm(decoded)
            velocity = self.out_proj(decoded) * torch.sigmoid(self.out_scale[:n_horizons]).unsqueeze(0)
            return velocity, attention_stack

        decoded = self.out_norm(self.decoder(query, memory))
        return self.out_proj(decoded) * torch.sigmoid(self.out_scale[:n_horizons]).unsqueeze(0)
