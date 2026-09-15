"""
VelocityTransformer: v_theta(z_tau, tau, c) trong OT-CFM (Section 4.3).
Non-autoregressive Transformer decoder - tất cả pred_len horizon query
được xử lý đồng thời, attend chung vào 1 memory chứa flow-time embedding
và context vector c.

FiLM conditioning theo từng horizon (film_gamma/film_beta) cho phép mỗi
query horizon áp affine transform riêng lên context, thay vì dùng chung
1 cách "chú ý" tới c cho mọi horizon.

out_proj khởi tạo zero + out_scale sigmoid(init~0.1) => velocity dự đoán
ban đầu gần 0, tránh bước cập nhật đầu tiên quá lớn làm hỏng gradient.
"""

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
        self.traj_embed = nn.Linear(2, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.step_emb = nn.Embedding(pred_len, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.cond_proj = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))

        self.film_gamma = nn.Embedding(pred_len, d_model)
        self.film_beta = nn.Embedding(pred_len, d_model)
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_beta.weight)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2)
        )
        self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding cho flow-time tau, giống chuẩn Transformer
        positional encoding nhưng dùng cho biến continuous [0,1]."""
        half = self.d_model // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.time_mlp(emb)

    def _decode_with_attn(self, x_emb: torch.Tensor, memory: torch.Tensor):
        """Chạy lại thủ công từng layer của self.decoder để lấy được
        cross-attention weight (average_attn_weights=True) — dùng cho
        phân tích XAI (Section 4.8), không dùng trong forward pass
        bình thường vì tốn thêm 1 chút overhead so với nn.TransformerDecoder
        có sẵn."""
        x = x_emb
        attn_per_layer = []
        for layer in self.decoder.layers:
            sa_out = layer.self_attn(
                layer.norm1(x), layer.norm1(x), layer.norm1(x), need_weights=False
            )[0]
            x = x + layer.dropout1(sa_out)

            normed = layer.norm2(x)
            mha_out, attn_w = layer.multihead_attn(
                normed, memory, memory, need_weights=True, average_attn_weights=True
            )
            attn_per_layer.append(attn_w)
            x = x + layer.dropout2(mha_out)

            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x)))))
            x = x + layer.dropout3(ff_out)

        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x, torch.stack(attn_per_layer, dim=0)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, return_attn: bool = False
    ):
        B, T, _ = x_t.shape
        step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
        x_emb = self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx)

        cond_vec = self.cond_proj(cond)

        gamma = self.film_gamma(step_idx[0]).unsqueeze(0)
        beta = self.film_beta(step_idx[0]).unsqueeze(0)
        x_emb = x_emb + (gamma * cond_vec.unsqueeze(1) + beta)

        memory = torch.cat([self._time_emb(t).unsqueeze(1), cond_vec.unsqueeze(1)], dim=1)
        if return_attn:
            dec_out, attn_stack = self._decode_with_attn(x_emb, memory)
            out = self.out_norm(dec_out)
            v = self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)
            return v, attn_stack
        out = self.out_norm(self.decoder(x_emb, memory))
        return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)
