from __future__ import annotations

import math

import numpy as np

ENV_FEATURE_DIMS: dict[str, int] = {
    "wind": 1,
    "intensity_class": 6,
    "move_velocity": 1,
    "velocity_history": 4,
    "rapid_intensification": 1,
    "month": 12,
    "location_lon_scs": 10,
    "location_lat_scs": 8,
    "bearing_to_scs_center": 16,
    "dist_to_scs_boundary": 5,
    "delta_velocity": 5,
    "history_direction12": 8,
    "history_direction24": 8,
    "history_inte_change24": 4,
    "gph500_mean": 1,
    "gph500_center": 1,
    "u500_mean": 1,
    "u500_center": 1,
    "v500_mean": 1,
    "v500_center": 1,
    "steering_speed": 1,
    "steering_dir_sin": 1,
    "steering_dir_cos": 1,
}
ENV_DIM_TOTAL = sum(ENV_FEATURE_DIMS.values())

_D3D_KEYS = {
    "gph500_mean",
    "gph500_center",
    "u500_mean",
    "u500_center",
    "v500_mean",
    "v500_center",
    "steering_speed",
    "steering_dir_sin",
    "steering_dir_cos",
}
ENV_1D_DIM = sum(d for k, d in ENV_FEATURE_DIMS.items() if k not in _D3D_KEYS)
ENV_3D_DIM = ENV_DIM_TOTAL - ENV_1D_DIM

_MOVE_VEL_NORM = 150.0
_UV500_NORM = 30.0
_GPH500_MEAN_M = 5880.0
_GPH500_STD_M = 150.0
_GPH500_MIN_M = 4000.0
_GPH500_MAX_M = 7000.0
_WIND_NORM_DENOM = 150.0
_INTENSITY_THRESHOLDS_MS = [17.2, 32.7, 41.5, 51.5, 65.0]

SCS_BBOX = dict(lon_min=100.0, lon_max=125.0, lat_min=5.0, lat_max=20.0)
SCS_CENTER = (112.5, 12.5)
SCS_DIAGONAL_KM = 3100.0
BOUNDARY_THRESHOLDS = [0.05, 0.15, 0.30]
DELTA_VEL_BINS = [-20.0, -5.0, 5.0, 20.0]


def _pos_onehot(val: float, lo: float, hi: float, n: int) -> list[int]:
    idx = int((val - lo) / (hi - lo) * n)
    v = [0] * n
    v[max(0, min(n - 1, idx))] = 1
    return v


def bearing_to_scs_center_onehot(lon_deg: float, lat_deg: float) -> list[int]:

    c_lon, c_lat = SCS_CENTER
    mid_lat = math.radians((lat_deg + c_lat) / 2.0)
    dx = (c_lon - lon_deg) * math.cos(mid_lat)
    dy = c_lat - lat_deg
    bearing = math.degrees(math.atan2(dx, dy)) % 360.0
    idx = int((bearing + 11.25) / 22.5) % 16
    v = [0] * 16
    v[idx] = 1
    return v


def dist_to_scs_boundary_onehot(lon_deg: float, lat_deg: float) -> list[int]:

    lo, hi = SCS_BBOX["lon_min"], SCS_BBOX["lon_max"]
    la, lb = SCS_BBOX["lat_min"], SCS_BBOX["lat_max"]
    if not (lo <= lon_deg <= hi and la <= lat_deg <= lb):
        return [1, 0, 0, 0, 0]
    d_min = min(
        (lon_deg - lo) * 111.0 * math.cos(math.radians(lat_deg)),
        (hi - lon_deg) * 111.0 * math.cos(math.radians(lat_deg)),
        (lat_deg - la) * 111.0,
        (lb - lat_deg) * 111.0,
    )
    r = d_min / SCS_DIAGONAL_KM
    if r < BOUNDARY_THRESHOLDS[0]:
        idx = 4
    elif r < BOUNDARY_THRESHOLDS[1]:
        idx = 3
    elif r < BOUNDARY_THRESHOLDS[2]:
        idx = 2
    else:
        idx = 1
    v = [0] * 5
    v[idx] = 1
    return v


def delta_velocity_onehot(delta_km_h: float) -> list[int]:

    bins = DELTA_VEL_BINS
    if delta_km_h <= bins[0]:
        idx = 0
    elif delta_km_h <= bins[1]:
        idx = 1
    elif delta_km_h <= bins[2]:
        idx = 2
    elif delta_km_h <= bins[3]:
        idx = 3
    else:
        idx = 4
    v = [0] * 5
    v[idx] = 1
    return v


def intensity_class_onehot(wind_ms: float) -> list[int]:

    idx = sum(wind_ms >= t for t in _INTENSITY_THRESHOLDS_MS)
    v = [0] * 6
    v[min(idx, 5)] = 1
    return v


def _read_uv500_from_npy(env_npy: dict, key: str) -> float:
    if not isinstance(env_npy, dict) or not env_npy.get("has_era5_data", True):
        return 0.0
    raw = env_npy.get(key, None)
    if raw is None:
        return 0.0
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if abs(v) > 2.0:
        return float(np.clip(v / _UV500_NORM, -1.0, 1.0))
    return float(np.clip(v, -1.0, 1.0))


def _read_scalar_from_npy(env_npy: dict, key: str, default: float = 0.0) -> float:
    if not isinstance(env_npy, dict):
        return default
    raw = env_npy.get(key, None)
    if raw is None:
        return default
    try:
        v = float(raw)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _read_vector_from_npy(env_npy: dict, key: str, dim: int) -> list[float]:
    if not isinstance(env_npy, dict):
        return [0.0] * dim
    raw = env_npy.get(key, None)
    if raw is None:
        return [0.0] * dim
    try:
        if hasattr(raw, "__iter__"):
            lst = [float(x) for x in raw]
            return lst[:dim] + [0.0] * (dim - len(lst))
        return [float(raw)] + [0.0] * (dim - 1)
    except (TypeError, ValueError):
        return [0.0] * dim


def build_env_features_one_step(
    lon_norm: float,
    lat_norm: float,
    wind_norm: float,
    timestamp: str,
    env_npy,
    prev_speed_kmh,
    pres_norm: float = 0.0,
) -> dict:

    lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
    lat_deg = (lat_norm * 50.0) / 10.0
    wind_kt = wind_norm * 25.0 + 40.0
    wind_ms = wind_kt * 0.5144

    feat: dict = {}

    if isinstance(env_npy, dict) and "wind" in env_npy:
        feat["wind"] = [float(env_npy["wind"])]
    else:
        feat["wind"] = [wind_ms / _WIND_NORM_DENOM]

    feat["intensity_class"] = intensity_class_onehot(wind_ms)

    mv_raw = 0.0
    if isinstance(env_npy, dict):
        v = env_npy.get("move_velocity", 0.0)
        if v is not None and v != -1:
            mv_raw = float(v)
            if 0.0 < mv_raw < 2.0:
                mv_raw = mv_raw * _MOVE_VEL_NORM
    feat["move_velocity"] = [mv_raw / _MOVE_VEL_NORM]

    vh = _read_vector_from_npy(env_npy, "velocity_history", 4)
    feat["velocity_history"] = [
        (
            float(np.clip(v / _MOVE_VEL_NORM, 0.0, 2.0))
            if abs(v) > 2.0
            else float(np.clip(v, 0.0, 2.0))
        )
        for v in vh
    ]

    ri = _read_scalar_from_npy(env_npy, "rapid_intensification", 0.0)
    feat["rapid_intensification"] = [float(ri > 0.5)]

    try:
        mi = int(timestamp[4:6]) - 1
    except Exception:
        mi = 0
    oh = [0] * 12
    oh[max(0, min(11, mi))] = 1
    feat["month"] = oh

    feat["location_lon_scs"] = _pos_onehot(lon_deg, 100.0, 125.0, 10)
    feat["location_lat_scs"] = _pos_onehot(lat_deg, 5.0, 25.0, 8)
    feat["bearing_to_scs_center"] = bearing_to_scs_center_onehot(lon_deg, lat_deg)
    feat["dist_to_scs_boundary"] = dist_to_scs_boundary_onehot(lon_deg, lat_deg)

    delta = (mv_raw - prev_speed_kmh) if prev_speed_kmh is not None else 0.0
    feat["delta_velocity"] = delta_velocity_onehot(delta)

    for key, dim in [("history_direction12", 8), ("history_direction24", 8)]:
        if isinstance(env_npy, dict) and key in env_npy:
            v = env_npy[key]
            v = list(v)[:dim] if hasattr(v, "__iter__") else [-1] * dim
            v = v + [0] * (dim - len(v))
            v = [-1] * dim if all(x == -1 for x in v) else v
        else:
            v = [-1] * dim
        feat[key] = v

    key = "history_inte_change24"
    if isinstance(env_npy, dict) and key in env_npy:
        v = env_npy[key]
        v = list(v)[:4] if hasattr(v, "__iter__") else [-1] * 4
        v = v + [0] * (4 - len(v))
        v = [-1] * 4 if all(x == -1 for x in v) else v
    else:
        v = [-1] * 4
    feat["history_inte_change24"] = v

    for feat_key, npy_key in [("gph500_mean", "gph500_mean"), ("gph500_center", "gph500_center")]:
        val = 0.0
        if isinstance(env_npy, dict) and env_npy.get("has_era5_data", False) and npy_key in env_npy:
            try:
                raw = float(env_npy[npy_key])
                if _GPH500_MIN_M <= raw <= _GPH500_MAX_M:
                    val = float(np.clip((raw - _GPH500_MEAN_M) / (_GPH500_STD_M + 1e-8), -3.0, 3.0))
                elif -3.0 <= raw <= 3.0:
                    val = float(np.clip(raw, -3.0, 3.0))
            except (TypeError, ValueError):
                val = 0.0
        feat[feat_key] = [val]

    for feat_key in ("u500_mean", "u500_center", "v500_mean", "v500_center"):
        val = 0.0
        if (
            isinstance(env_npy, dict)
            and env_npy.get("has_era5_data", False)
            and feat_key in env_npy
        ):
            val = _read_uv500_from_npy(env_npy, feat_key)
        feat[feat_key] = [val]

    st_speed = _read_scalar_from_npy(env_npy, "steering_speed", 0.0)
    st_sin = _read_scalar_from_npy(env_npy, "steering_dir_sin", 0.0)
    st_cos = _read_scalar_from_npy(env_npy, "steering_dir_cos", 1.0)

    if st_speed == 0.0 and isinstance(env_npy, dict):
        u, v = feat["u500_mean"][0], feat["v500_mean"][0]
        mag = math.sqrt(u * u + v * v)
        if mag > 1e-4:
            st_speed = float(np.clip(mag, 0.0, 2.0))
            st_sin = float(v / mag)
            st_cos = float(u / mag)

    feat["steering_speed"] = [float(np.clip(st_speed, 0.0, 2.0))]
    feat["steering_dir_sin"] = [float(np.clip(st_sin, -1.0, 1.0))]
    feat["steering_dir_cos"] = [float(np.clip(st_cos, -1.0, 1.0))]

    return feat
