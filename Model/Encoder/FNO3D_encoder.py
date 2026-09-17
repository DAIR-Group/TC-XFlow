from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_time: int = 4,
        modes_lat: int = 4,
        modes_lon: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_time = modes_time
        self.modes_lat = modes_lat
        self.modes_lon = modes_lon

        scale = 1.0 / math.sqrt(in_channels * out_channels)
        weight_shape = (in_channels, out_channels, modes_time, modes_lat, modes_lon)

       
        for quadrant in range(1, 5):
            setattr(self, f"weight_real_{quadrant}", nn.Parameter(scale * torch.randn(*weight_shape)))
            setattr(self, f"weight_imag_{quadrant}", nn.Parameter(scale * torch.randn(*weight_shape)))

    def _complex_multiply(self, x: torch.Tensor, weight_real: torch.Tensor, weight_imag: torch.Tensor):
        weight_real = weight_real.float()
        weight_imag = weight_imag.float()
        x_real = x.real.float()
        x_imag = x.imag.float()
        out_real = torch.einsum("bipqr,ijpqr->bjpqr", x_real, weight_real) - torch.einsum(
            "bipqr,ijpqr->bjpqr", x_imag, weight_imag
        )
        out_imag = torch.einsum("bipqr,ijpqr->bjpqr", x_real, weight_imag) + torch.einsum(
            "bipqr,ijpqr->bjpqr", x_imag, weight_real
        )
        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, n_time, n_lat, n_lon = x.shape
        x_freq = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")

        out_freq = torch.zeros(
            batch_size, self.out_channels, n_time, n_lat, n_lon // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        mt = min(self.modes_time, n_time // 2)
        ml = min(self.modes_lat, n_lat // 2)
        mo = min(self.modes_lon, n_lon // 2 + 1)

    
        out_freq[:, :, :mt, :ml, :mo] = self._complex_multiply(
            x_freq[:, :, :mt, :ml, :mo], self.weight_real_1, self.weight_imag_1
        )
   
        out_freq[:, :, -mt:, :ml, :mo] = self._complex_multiply(
            x_freq[:, :, -mt:, :ml, :mo], self.weight_real_2, self.weight_imag_2
        )
  
        out_freq[:, :, :mt, -ml:, :mo] = self._complex_multiply(
            x_freq[:, :, :mt, -ml:, :mo], self.weight_real_3, self.weight_imag_3
        )
  
        out_freq[:, :, -mt:, -ml:, :mo] = self._complex_multiply(
            x_freq[:, :, -mt:, -ml:, :mo], self.weight_real_4, self.weight_imag_4
        )

        return torch.fft.irfftn(out_freq, s=(n_time, n_lat, n_lon), dim=(-3, -2, -1), norm="ortho").to(
            x.dtype
        )


class FNOLayer3d(nn.Module):


    def __init__(
        self,
        channels: int,
        modes_time: int = 4,
        modes_lat: int = 4,
        modes_lon: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spectral_conv = SpectralConv3d(channels, channels, modes_time, modes_lat, modes_lon)
        self.pointwise_residual = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.activation = nn.GELU()
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(
            self.activation(self.norm(self.spectral_conv(x) + self.pointwise_residual(x)))
        )


class FNO3DEncoder(nn.Module):


    def __init__(
        self,
        in_channel: int = 13,
        out_channel: int = 1,
        d_model: int = 32,
        n_layers: int = 4,
        modes_t: int = 4,
        modes_h: int = 4,
        modes_w: int = 4,
        spatial_down: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.spatial_down = spatial_down
        self.d_model = d_model
        self.in_channel = in_channel

        self.lift = nn.Sequential(
            nn.Conv3d(in_channel, d_model, kernel_size=1, bias=False),
            nn.InstanceNorm3d(d_model, affine=True),
            nn.GELU(),
        )

        self.fno_layers = nn.ModuleList(
            [FNOLayer3d(d_model, modes_t, modes_h, modes_w, dropout) for _ in range(n_layers)]
        )

        self.bottleneck_proj = nn.Conv3d(d_model, 128, kernel_size=1, bias=False)

        self.summary_head = nn.Sequential(
            nn.Conv3d(d_model, 16, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(16, out_channel, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )

    def _downsample_spatial(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_time, n_lat, n_lon = x.shape
        if n_lat == self.spatial_down and n_lon == self.spatial_down:
            return x
        x_flat_time = x.permute(0, 2, 1, 3, 4).reshape(batch_size * n_time, n_channels, n_lat, n_lon)
        x_flat_time = F.interpolate(
            x_flat_time, size=(self.spatial_down, self.spatial_down), mode="bilinear", align_corners=False
        )
        return x_flat_time.reshape(
            batch_size, n_time, n_channels, self.spatial_down, self.spatial_down
        ).permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, summary = self.encode(x)
        return summary

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        if x.dim() == 4:
            x = x.unsqueeze(1)

        _, n_channels, _, _, _ = x.shape

        if n_channels == 1 and self.in_channel != 1:
            x = x.expand(-1, self.in_channel, -1, -1, -1)

        x = self._downsample_spatial(x)
        x = self.lift(x)

        for layer in self.fno_layers:
            x = layer(x)

        bottleneck = self.bottleneck_proj(x)
        bottleneck = F.adaptive_avg_pool3d(bottleneck, (None, 4, 4))

        summary = self.summary_head(x)

        return bottleneck, summary
