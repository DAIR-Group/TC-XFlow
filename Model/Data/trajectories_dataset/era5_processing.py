from __future__ import annotations

import logging
import os

import numpy as np
import torch

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import netCDF4 as nc

    HAS_NC = True
except ImportError:
    HAS_NC = False

logger = logging.getLogger(__name__)

ERA5_H = 81
ERA5_W = 81
ERA5_CH = 13

ERA5_MEAN = np.array(
    [
        12444.037,
        5844.935,
        1482.430,
        752.466,
        -1.052,
        -0.188,
        -0.407,
        -0.888,
        -0.055,
        1.655,
        1.343,
        1.018,
        301.133,
    ],
    dtype=np.float32,
)
ERA5_STD = np.array(
    [
        113.583,
        57.135,
        37.852,
        37.274,
        13.315,
        8.042,
        7.911,
        7.494,
        8.377,
        5.999,
        6.203,
        6.555,
        3.023,
    ],
    dtype=np.float32,
)

_SENTINEL_LARGE = 20000.0
_ZERO_SENTINEL_CHANNELS = {0}
_GPH_VALID_MIN = 25.0
_GPH_VALID_MAX = 95.0
_SST_CHANNEL = 12
_SST_VALID_MIN = 270.0
_SST_FILL_K = 298.0


def normalize_era5_patch(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    for c in range(ERA5_CH):
        ch = arr[:, :, c]
        ch[ch > _SENTINEL_LARGE] = np.nan
        if c < 4:
            ch[ch < 0] = np.nan
        if c in _ZERO_SENTINEL_CHANNELS:
            ch[ch == 0.0] = np.nan
        if c == _SST_CHANNEL:
            ch[ch < _SST_VALID_MIN] = _SST_FILL_K
        if np.any(np.isnan(ch)):
            valid_vals = ch[~np.isnan(ch)]
            fill_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else float(ERA5_MEAN[c])
            ch[np.isnan(ch)] = fill_val
        arr[:, :, c] = (ch - ERA5_MEAN[c]) / (ERA5_STD[c] + 1e-6)
    return np.clip(arr, -5.0, 5.0)


def load_era5_file(path: str) -> np.ndarray | None:
    try:
        if path.endswith(".npy"):
            arr = np.load(path).astype(np.float32)
        elif path.endswith(".nc") and HAS_NC:
            with nc.Dataset(path) as ds:
                keys = list(ds.variables.keys())
                arr = np.array(ds.variables[keys[-1]][:]).astype(np.float32)
        else:
            return None

        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        if arr.ndim != 3:
            return None

        if arr.shape[0] == ERA5_CH:
            arr = arr.transpose(1, 2, 0)

        H, W, C = arr.shape
        if H != ERA5_H or W != ERA5_W:
            if HAS_CV2:
                arr = cv2.resize(arr, (ERA5_W, ERA5_H))
            else:
                arr = arr[:ERA5_H, :ERA5_W, :]
                if arr.shape[0] < ERA5_H:
                    arr = np.pad(arr, ((0, ERA5_H - arr.shape[0]), (0, 0), (0, 0)))
                if arr.shape[1] < ERA5_W:
                    arr = np.pad(arr, ((0, 0), (0, ERA5_W - arr.shape[1]), (0, 0)))

        if arr.shape[2] < ERA5_CH:
            pad_ch = np.zeros((ERA5_H, ERA5_W, ERA5_CH - arr.shape[2]), dtype=np.float32)
            arr = np.concatenate([arr, pad_ch], axis=2)
        arr = arr[:, :, :ERA5_CH]

        return normalize_era5_patch(arr)
    except Exception as e:
        logger.debug(f"ERA5 file load error {path}: {e}")
        return None

def read_era5_step(era5_data_path: str, year, ty_name, timestamp) -> torch.Tensor:
    folder = os.path.join(era5_data_path, str(year), str(ty_name))
    if not os.path.exists(folder):
        return torch.zeros(ERA5_H, ERA5_W, ERA5_CH)

    prefix = f"WP{year}{ty_name}_{timestamp}"
    for ext in (".npy", ".nc"):
        p = os.path.join(folder, prefix + ext)
        if os.path.exists(p):
            arr = load_era5_file(p)
            if arr is not None:
                return torch.from_numpy(arr).float()

    try:
        for fname in sorted(os.listdir(folder)):
            if timestamp in fname and fname.endswith((".npy", ".nc")):
                arr = load_era5_file(os.path.join(folder, fname))
                if arr is not None:
                    return torch.from_numpy(arr).float()
    except Exception:
        pass

    return torch.zeros(ERA5_H, ERA5_W, ERA5_CH)
