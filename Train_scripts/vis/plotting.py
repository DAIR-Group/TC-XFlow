from __future__ import annotations

import re

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from vis.style import STYLE, seed_color
from vis.map_render import make_map_ax


def infer_seed_label(ckpt_path: str, idx: int) -> str:
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "seed" in ck:
            return str(ck["seed"])
    except Exception:
        pass
    m = re.search(r"seed[_-]?(\d+)", ckpt_path)
    return m.group(1) if m else str(idx)


def plot_multi_seed_comparison(
    obs_deg,
    gt_deg,
    preds_by_seed,
    errors_by_seed,
    t_name: str,
    output_path: str,
    winds_pred_by_seed: dict | None = None,
    wind_gt=None,
):
    transform = ccrs.PlateCarree() if HAS_CARTOPY else None
    cur_pos = obs_deg[-1]

    all_deg = np.vstack([obs_deg, gt_deg] + list(preds_by_seed.values()))
    lon_span = all_deg[:, 0].max() - all_deg[:, 0].min()
    lat_span = all_deg[:, 1].max() - all_deg[:, 1].min()
    margin_lon = float(np.clip(lon_span * 0.10, 1.0, 4.5)) + max(0.0, (lat_span - lon_span) * 0.35)
    margin_lat = float(np.clip(lat_span * 0.10, 1.0, 4.5))
    lon_range = (all_deg[:, 0].min() - margin_lon, all_deg[:, 0].max() + margin_lon)
    lat_range = (all_deg[:, 1].min() - margin_lat, all_deg[:, 1].max() + margin_lat)

    map_aspect = (lon_range[1] - lon_range[0]) / max(lat_range[1] - lat_range[0], 0.01)
    fig_h = 10.0
    fig_w_map = float(np.clip(fig_h * map_aspect, 5.0, 13.0))

    has_wind = bool(winds_pred_by_seed) and wind_gt is not None
    if has_wind:
        fig_w_wind = 6.0
        fig = plt.figure(figsize=(fig_w_map + fig_w_wind + 1.0, fig_h), facecolor=STYLE["bg_color"])
        gs = fig.add_gridspec(1, 2, width_ratios=[fig_w_map, fig_w_wind], wspace=0.15)
        ax = make_map_ax(fig, gs[0, 0], lon_range, lat_range)
        ax_wind = fig.add_subplot(gs[0, 1])
        ax_wind.set_facecolor(STYLE["bg_color"])
    else:
        fig = plt.figure(figsize=(fig_w_map, fig_h), facecolor=STYLE["bg_color"])
        ax = make_map_ax(fig, 111, lon_range, lat_range)

    def plot_line(x, y, **kw):
        if HAS_CARTOPY:
            ax.plot(x, y, transform=transform, **kw)
        else:
            ax.plot(x, y, **kw)

    plot_line(
        obs_deg[:, 0],
        obs_deg[:, 1],
        marker="o",
        color=STYLE["obs_color"],
        linewidth=STYLE["lw_thin"],
        markersize=STYLE["marker_size"],
        zorder=6,
        label="Observed",
    )

    gt_lon = np.concatenate([[cur_pos[0]], gt_deg[:, 0]])
    gt_lat = np.concatenate([[cur_pos[1]], gt_deg[:, 1]])
    plot_line(
        gt_lon,
        gt_lat,
        marker="o",
        color=STYLE["gt_color"],
        linewidth=2.2,
        markersize=STYLE["marker_size"] + 1,
        zorder=10,
        label="Actual Track",
    )

    handles = [
        Line2D([0], [0], color=STYLE["obs_color"], marker="o", lw=1.2, label="Observed"),
        Line2D([0], [0], color=STYLE["gt_color"], marker="o", lw=2.2, label="Actual Track"),
    ]

    wind_lines = ["Wind MAE (kt):"] if has_wind else []
    for idx, (seed_label, pred_deg) in enumerate(preds_by_seed.items()):
        color = seed_color(seed_label, idx)
        pred_lon = np.concatenate([[cur_pos[0]], pred_deg[:, 0]])
        pred_lat = np.concatenate([[cur_pos[1]], pred_deg[:, 1]])
        plot_line(
            pred_lon,
            pred_lat,
            marker="o",
            color=color,
            linewidth=STYLE["lw_main"],
            markersize=STYLE["marker_size"] - 1,
            zorder=9,
            alpha=0.9,
        )
        mean_dpe = errors_by_seed[seed_label].mean()
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                lw=1.6,
                label=f"seed={seed_label} (Mean DPE={mean_dpe:.0f}km)",
            )
        )

        if has_wind:
            wpred = winds_pred_by_seed.get(seed_label)
            if wpred is not None:
                T = min(len(wpred), len(wind_gt))
                hours = np.arange(1, T + 1) * 6
                ax_wind.plot(
                    hours,
                    wpred[:T],
                    "o-",
                    color=color,
                    linewidth=1.6,
                    markersize=3.5,
                    label=f"seed={seed_label}",
                )
                mae = float(np.abs(wpred[:T] - wind_gt[:T]).mean())
                wind_lines.append(f" seed={seed_label}: {mae:.1f} kt")

    if has_wind:
        T_gt = len(wind_gt)
        hours_gt = np.arange(1, T_gt + 1) * 6
        ax_wind.plot(
            hours_gt,
            wind_gt,
            "o-",
            color=STYLE["gt_color"],
            linewidth=2.2,
            markersize=4,
            label="Actual",
            zorder=10,
        )
        ax_wind.set_xlabel("Forecast Lead Time (h)", fontsize=9)
        ax_wind.set_ylabel("Wind Speed (kt)", fontsize=9)
        ax_wind.set_title("Wind Speed Comparison", fontsize=11, fontweight="bold")
        ax_wind.legend(fontsize=7.5, framealpha=0.9)
        ax_wind.grid(True, alpha=0.3, linestyle="--")

        ax.text(
            0.02,
            0.03,
            "\n".join(wind_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            color=STYLE["text_color"],
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec=STYLE["panel_edge"], lw=0.8
            ),
            zorder=16,
        )

    ax.set_title(
        f"{t_name} - Seed Comparison", fontsize=13, fontweight="bold", color=STYLE["text_color"]
    )
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {output_path}")
