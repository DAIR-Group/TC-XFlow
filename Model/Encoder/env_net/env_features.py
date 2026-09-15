from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from Model.Encoder.env_net import (
    build_env_features_one_step,
    ENV_FEATURE_DIMS,
)

_SST_VALID_MIN = 270.0
_GPH_VALID_MIN = 25.0
_GPH_VALID_MAX = 95.0
_SST_FILL_K = 298.0


def env_data_processing(env_dict: dict) -> dict:
    if not isinstance(env_dict, dict):
        return {}

    already_normed = bool(env_dict.get("gph500_already_normed", False))
    skip_keys = (
        "gph500_already_normed",
        "has_era5_data",
        "gph500_mean_already_normed",
        "gph500_center_already_normed",
    )

    cleaned = {"gph500_already_normed": already_normed}
    for k, v in env_dict.items():
        if k in skip_keys:
            continue
        if isinstance(v, (list, np.ndarray)):
            cleaned[k] = v
        elif isinstance(v, bool):
            cleaned[k] = v
        elif v == -1:
            cleaned[k] = 0.0
        else:
            cleaned[k] = v

    cleaned["has_era5_data"] = env_dict.get("has_era5_data", True)

    for sst_key in ("sst_mean", "sst_center", "sst"):
        if sst_key in cleaned:
            val = cleaned[sst_key]
            if val is None or val == 0 or (isinstance(val, float) and val < _SST_VALID_MIN):
                cleaned[sst_key] = _SST_FILL_K

    if not already_normed:
        for gph_key in ("gph500_mean", "gph500_center"):
            if gph_key in cleaned:
                val = cleaned[gph_key]
                if val is not None and isinstance(val, (int, float)):
                    if val < _GPH_VALID_MIN or val > _GPH_VALID_MAX:
                        cleaned[gph_key] = None

    return cleaned


def compute_env_features(load_env_fn, year, ty_name, dates, obs_traj, obs_Me) -> dict:
    T = len(dates)
    all_feats = []
    prev_speed = None

    for t in range(T):
        env_npy = load_env_fn(year, ty_name, dates[t])
        feat = build_env_features_one_step(
            lon_norm=float(obs_traj[0, t]),
            lat_norm=float(obs_traj[1, t]),
            wind_norm=float(obs_Me[1, t]),
            pres_norm=float(obs_Me[0, t]),
            timestamp=dates[t],
            env_npy=env_npy,
            prev_speed_kmh=prev_speed,
        )
        all_feats.append(feat)
        if isinstance(env_npy, dict):
            mv = float(env_npy.get("move_velocity", 0.0) or 0.0)
            prev_speed = mv if mv != -1 else 0.0

    env_out = {}
    for key, dim in ENV_FEATURE_DIMS.items():
        rows = []
        for feat in all_feats:
            v = feat.get(key, [0.0] * dim)
            t = torch.tensor(v, dtype=torch.float)
            if t.numel() < dim:
                t = F.pad(t, (0, dim - t.numel()))
            rows.append(t[:dim])
        env_out[key] = torch.stack(rows, dim=0)
    return env_out


def embed_time(date_list) -> torch.Tensor:
    rows = []
    for d in date_list:
        try:
            rows.append(
                [
                    (float(d[:4]) - 1949) / 70.0 - 0.5,
                    (float(d[4:6]) - 1) / 11.0 - 0.5,
                    (float(d[6:8]) - 1) / 30.0 - 0.5,
                    float(d[8:10]) / 18.0 - 0.5,
                ]
            )
        except Exception:
            rows.append([0.0, 0.0, 0.0, 0.0])
    return torch.tensor(rows, dtype=torch.float).t().unsqueeze(0)
