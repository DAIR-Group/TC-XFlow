from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_MOVE_VEL_NORM = 150.0
_UV500_NORM = 30.0

_SCS_LON_MIN, _SCS_LON_MAX = 99.0, 121.0
_SCS_LAT_MIN, _SCS_LAT_MAX = 0.0, 23.0
_SCS_MIN_PCT_DEFAULT = 15.0


def build_csv_env_lookup(csv_path: str) -> dict:
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas is not available — CSV fallback is unavailable")
        return {}

    if not os.path.exists(csv_path):
        logger.warning(f"CSV fallback not found: {csv_path}")
        return {}

    logger.info(f"Loading CSV env fallback: {csv_path}")
    try:
        df = pd.read_csv(csv_path, dtype={"storm_name": str, "dt": str})
    except Exception as e:
        logger.warning(f"CSV load failed: {e}")
        return {}

    lookup: dict = {}
    for _, row in df.iterrows():
        yr = str(int(float(row["year"])))
        name_raw = str(row["storm_name"])
        name_strip = name_raw.lstrip("0") or "0"
        ts = str(row["dt"])

        dir12 = [float(row.get(f"env_dir12_{i}", 0.0)) for i in range(8)]
        dir24 = [float(row.get(f"env_dir24_{i}", 0.0)) for i in range(8)]
        inten24 = [float(row.get(f"env_inten24_{i}", 0.0)) for i in range(4)]

        gph_mean = float(row.get("env_gph500_mean", -29.5))
        gph_center = float(row.get("env_gph500_center", -29.5))

        mv_norm = float(row.get("env_move_velocity", 0.0))
        mv_raw = mv_norm * _MOVE_VEL_NORM

        def _norm_uv(col_name):
            raw = float(row.get(col_name, 0.0))
            if raw <= 0.0 or raw > 50000.0:
                return 0.0
            return float(np.clip(raw / _UV500_NORM, -1.0, 1.0))

        u500_mean = _norm_uv("d3d_u500_mean_raw")
        u500_center = (
            _norm_uv("d3d_u500_center_raw") if "d3d_u500_center_raw" in df.columns else u500_mean
        )
        v500_mean = _norm_uv("d3d_v500_mean_raw")
        v500_center = (
            _norm_uv("d3d_v500_center_raw") if "d3d_v500_center_raw" in df.columns else v500_mean
        )

        entry = {
            "gph500_mean": gph_mean,
            "gph500_center": gph_center,
            "gph500_already_normed": True,
            "u500_mean": u500_mean,
            "u500_center": u500_center,
            "v500_mean": v500_mean,
            "v500_center": v500_center,
            "has_era5_data": (u500_mean != 0.0),
            "move_velocity": mv_raw,
            "history_direction12": dir12,
            "history_direction24": dir24,
            "history_inte_change24": inten24,
        }

        for name_try in (name_raw, name_strip, name_raw.zfill(4), name_raw.zfill(2)):
            lookup[(yr, name_try, ts)] = entry

    n_storms = df["storm_name"].nunique()
    logger.info(f"CSV env lookup built: {len(lookup)} entries ({n_storms} storms)")

    sample_u = [v["u500_mean"] for v in list(lookup.values())[:100]]
    nonzero = sum(1 for x in sample_u if x != 0.0)
    logger.info(
        f"CSV u500_mean: {nonzero}/100 non-zero in first 100 entries "
        f"(mean={np.mean(sample_u):.4f})"
    )

    return lookup


def auto_discover_csv(root_path: str) -> str | None:
    candidates = [
        os.path.join(root_path, "all_storms_final.csv"),
        os.path.join(os.path.dirname(root_path), "all_storms_final.csv"),
        os.path.join(os.path.dirname(os.path.dirname(root_path)), "all_storms_final.csv"),
        "/kaggle/input/datasets/gmnguynhng/data-tc-finall/all_storms_final.csv",
        "/kaggle/working/all_storms_final.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            logger.info(f"Auto-discovered CSV: {p}")
            return p
    return None


def auto_discover_synthetic_csv(root_path: str) -> str | None:
    candidates = [
        os.path.join(root_path, "all_storms_synthetic.csv"),
        os.path.join(os.path.dirname(root_path), "all_storms_synthetic.csv"),
        os.path.join(os.path.dirname(os.path.dirname(root_path)), "all_storms_synthetic.csv"),
        "/kaggle/working/all_storms_synthetic.csv",
        "/kaggle/input/datasets/gmnguynhng/data-tc-finall/all_storms_synthetic.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            logger.info(f"Auto-discovered synthetic CSV: {p}")
            return p
    return None


def storm_touches_scs_vietnam(filepath: str, min_pct: float = _SCS_MIN_PCT_DEFAULT) -> bool:
    try:
        lons, lats = [], []
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "//", "-", "=")):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    int(parts[0])
                except ValueError:
                    continue
                try:
                    lon_norm = float(parts[1])
                    lat_norm = float(parts[2])
                except (ValueError, IndexError):
                    continue
                lons.append((lon_norm * 50.0 + 1800.0) / 10.0)
                lats.append((lat_norm * 50.0) / 10.0)

        if not lons:
            return True

        n_in_box = sum(
            1
            for lon, lat in zip(lons, lats)
            if _SCS_LON_MIN <= lon <= _SCS_LON_MAX and _SCS_LAT_MIN <= lat <= _SCS_LAT_MAX
        )
        pct_in_box = 100.0 * n_in_box / len(lons)
        return pct_in_box >= min_pct
    except Exception as e:
        logger.warning(
            f"storm_touches_scsfailed on {filepath}: "
            f"{e} — keeping file (fail-open, not fail-closed)."
        )
        return True
