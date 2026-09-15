from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Model.Data.trajectories_dataset.csv_lookup import (
    build_csv_env_lookup,
    auto_discover_csv,
    auto_discover_synthetic_csv,
    storm_touches_scs_vietnam,
    _SCS_LON_MIN,
    _SCS_LON_MAX,
    _SCS_LAT_MIN,
    _SCS_LAT_MAX,
    _SCS_MIN_PCT_DEFAULT,
)
from Model.Encoder.env_net.env_features import env_data_processing, compute_env_features, embed_time
from Model.Data.trajectories_dataset.era5_processing import ERA5_H, ERA5_W, ERA5_CH, read_era5_step

logger = logging.getLogger(__name__)

_LON_VALID_MIN = 73.00
_LON_VALID_MAX = 186.40
_LAT_VALID_MAX = 64.00
_LON_NORM_MIN = -21.4000
_LON_NORM_MAX = 1.2800
_LAT_NORM_MIN = 0.0000
_LAT_NORM_MAX = 12.8000

_NPY_KEY_REMAP = {"gph500_mean_n": "gph500_mean", "gph500_center_n": "gph500_center"}
_NPY_U500_KEY_REMAP = {
    "u500_mean_n": "u500_mean",
    "u500_center_n": "u500_center",
    "v500_mean_n": "v500_mean",
    "v500_center_n": "v500_center",
}


def seq_collate(data):
    (
        obs_traj,
        pred_traj,
        obs_rel,
        pred_rel,
        nlp,
        mask,
        obs_Me,
        pred_Me,
        obs_Me_rel,
        pred_Me_rel,
        obs_date,
        pred_date,
        img_obs,
        img_pred,
        env_data_raw,
        tyID,
    ) = zip(*data)

    def traj_TBC(lst):
        return torch.cat(lst, dim=0).permute(2, 0, 1)

    obs_traj_out = traj_TBC(obs_traj)
    pred_traj_out = traj_TBC(pred_traj)
    obs_rel_out = traj_TBC(obs_rel)
    pred_rel_out = traj_TBC(pred_rel)
    obs_Me_out = traj_TBC(obs_Me)
    pred_Me_out = traj_TBC(pred_Me)
    obs_Me_rel_out = traj_TBC(obs_Me_rel)
    pred_Me_rel_out = traj_TBC(pred_Me_rel)

    nlp_out = torch.tensor(
        [v for sl in nlp for v in (sl if hasattr(sl, "__iter__") else [sl])], dtype=torch.float
    )
    mask_out = torch.cat(list(mask), dim=0).permute(1, 0)

    counts = torch.tensor([t.shape[0] for t in obs_traj])
    cum = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.tensor([0]), cum[:-1]])
    seq_start_end = torch.stack([starts, cum], dim=1)

    img_obs_out = torch.stack(list(img_obs), dim=0).permute(0, 4, 1, 2, 3).float()
    img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()

    env_out = _collate_env(env_data_raw)

    return (
        obs_traj_out,
        pred_traj_out,
        obs_rel_out,
        pred_rel_out,
        nlp_out,
        mask_out,
        seq_start_end,
        obs_Me_out,
        pred_Me_out,
        obs_Me_rel_out,
        pred_Me_rel_out,
        img_obs_out,
        img_pred_out,
        env_out,
        None,
        list(tyID),
    )


def _collate_env(env_data_raw):
    valid_envs = [d for d in env_data_raw if isinstance(d, dict)]
    if not valid_envs:
        return None

    skip_keys = {
        "gph500_already_normed",
        "has_era5_data",
        "gph500_mean_already_normed",
        "gph500_center_already_normed",
        "history_direction12_valid",
        "history_direction24_valid",
        "history_inte_change24_valid",
    }
    all_keys = set()
    for d in valid_envs:
        all_keys.update(d.keys())
    all_keys -= skip_keys

    env_out = {}
    for key in sorted(all_keys):
        vals = []
        for d in env_data_raw:
            if isinstance(d, dict) and key in d:
                v = d[key]
                v = torch.tensor(v, dtype=torch.float) if not torch.is_tensor(v) else v.float()
                vals.append(v)
            else:
                ref = next((d[key] for d in valid_envs if key in d), None)
                if ref is not None:
                    rt = (
                        torch.tensor(ref, dtype=torch.float)
                        if not torch.is_tensor(ref)
                        else ref.float()
                    )
                    vals.append(torch.zeros_like(rt))
                else:
                    vals.append(torch.zeros(1))
        try:
            env_out[key] = torch.stack(vals, dim=0)
        except Exception:
            try:
                mx = max(v.numel() for v in vals)
                padded = [F.pad(v.flatten(), (0, mx - v.numel())) for v in vals]
                env_out[key] = torch.stack(padded, dim=0)
            except Exception:
                pass
    return env_out


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim=" ",
        other_modal="gph",
        test_year=None,
        type="train",
        split=None,
        is_test=False,
        csv_env_path=None,
        synthetic_csv_path=None,
        filter_region=False,
        min_pct_in_scs=_SCS_MIN_PCT_DEFAULT,
        **kwargs,
    ):
        super().__init__()

        dtype = split if split is not None else type
        if isinstance(data_dir, dict):
            root = data_dir["root"]
            dtype = data_dir.get("type", dtype)
        else:
            root = data_dir
        if is_test and dtype not in ("val", "test"):
            dtype = "test"

        self._resolve_paths(root, dtype)
        self._is_train = dtype == "train"

        self._csv_env_lookup = self._load_csv_env_lookup(csv_env_path)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.modal_name = other_modal

        self._synthetic_lookup = {}
        if self._is_train:
            if synthetic_csv_path is None:
                synthetic_csv_path = auto_discover_synthetic_csv(self.root_path)
            if synthetic_csv_path:
                self._synthetic_lookup = self._build_synthetic_lookup(synthetic_csv_path)
                logger.info(f"Loaded synthetic: {len(self._synthetic_lookup)} storms")

        if not os.path.exists(self.best_track_data_path):
            logger.error(f"Missing Data1d: {self.best_track_data_path}")
            self.num_seq = 0
            self.seq_start_end = []
            self.tyID = []
            return

        self._load_real_sequences(
            test_year, filter_region, min_pct_in_scs, min_ped, threshold, obs_len, pred_len, delim
        )

        real_seqs = len(self.tyID)
        logger.info(
            f"Loaded {real_seqs} real sequences (filtered "
            f"{self._n_filtered_coord} outlier coords)"
        )

        if self._is_train and self._synthetic_lookup:
            self._load_synthetic_sequences(min_ped, threshold, obs_len, pred_len, delim)
            logger.info(f"Total sequences after synthetic: {len(self.tyID)}")

        self.num_seq = len(self.tyID)
        cum = np.cumsum(self._num_peds_in_seq).tolist()
        self.seq_start_end = list(zip([0] + cum[:-1], cum))

    def _resolve_paths(self, root, dtype):
        root = os.path.abspath(root)
        bn = os.path.basename(root)
        if bn in ("train", "test", "val"):
            self.root_path = os.path.dirname(os.path.dirname(root))
        elif bn == "Data1d":
            self.root_path = os.path.dirname(root)
        else:
            self.root_path = root

        self.best_track_data_path = os.path.join(self.root_path, "Data1d", dtype)
        self.era5_data_path = os.path.join(self.root_path, "Data3d")
        for env_name in ("Env_data", "ENV_DATA", "env_data", "Env_Data"):
            candidate = os.path.join(self.root_path, env_name)
            if os.path.exists(candidate):
                self.env_path = candidate
                break
        else:
            self.env_path = os.path.join(self.root_path, "Env_data")

    def _load_csv_env_lookup(self, csv_env_path) -> dict:
        env_path_missing = not os.path.exists(self.env_path)
        lookup = {}
        if env_path_missing or csv_env_path is not None:
            path = csv_env_path or auto_discover_csv(self.root_path)
            if path:
                lookup = build_csv_env_lookup(path)
        if not lookup:
            auto_csv = auto_discover_csv(self.root_path)
            if auto_csv:
                lookup = build_csv_env_lookup(auto_csv)
        return lookup

    def _load_real_sequences(
        self, test_year, filter_region, min_pct_in_scs, min_ped, threshold, obs_len, pred_len, delim
    ):
        all_files = [
            os.path.join(self.best_track_data_path, f)
            for f in sorted(os.listdir(self.best_track_data_path))
            if f.endswith(".txt") and (test_year is None or str(test_year) in f)
        ]

        if filter_region:
            n_before = len(all_files)
            all_files = [
                f for f in all_files if storm_touches_scs_vietnam(f, min_pct=min_pct_in_scs)
            ]
            logger.info(
                f"filter_region=True (min_pct={min_pct_in_scs}%): kept "
                f"{len(all_files)}/{n_before} storm files whose track has >= "
                f"{min_pct_in_scs}% of points in the South China Sea / "
                f"(lon {_SCS_LON_MIN}-{_SCS_LON_MAX}E, "
                f"lat {_SCS_LAT_MIN}-{_SCS_LAT_MAX}N)"
            )

        self.obs_traj_raw, self.pred_traj_raw = [], []
        self.obs_Me_raw, self.pred_Me_raw = [], []
        self.obs_rel_raw, self.pred_rel_raw = [], []
        self.obs_Me_rel_raw, self.pred_Me_rel_raw = [], []
        self.non_linear_ped = []
        self.tyID = []
        self._num_peds_in_seq = []
        self.env_cache: dict = {}
        self._n_filtered_coord = 0

        for path in all_files:
            self._process_storm_file(path, min_ped, threshold, obs_len, pred_len, delim, self.skip)

    def _process_storm_file(self, path, min_ped, threshold, obs_len, pred_len, delim, skip):
        base = os.path.splitext(os.path.basename(path))[0]
        parts = base.split("_")
        f_year = parts[0] if parts else "unknown"
        f_name = "_".join(parts[1:]) if len(parts) > 1 else base

        d = self._read_file(path, delim)
        data = d["main"]
        add = d["addition"]
        if len(data) < self.seq_len:
            return

        frames = np.unique(data[:, 0]).tolist()
        frame_data = [data[data[:, 0] == f] for f in frames]
        n_frames = len(frames)
        if n_frames < self.seq_len:
            return
        n_seq = (n_frames - self.seq_len) // skip + 1

        for idx in range(0, n_seq * skip, skip):
            if idx + self.seq_len > len(frame_data):
                break
            self._process_window(
                frame_data, idx, f_year, f_name, add, min_ped, threshold, obs_len, pred_len
            )

    def _process_window(
        self, frame_data, idx, f_year, f_name, add, min_ped, threshold, obs_len, pred_len
    ):
        seg = np.concatenate(frame_data[idx : idx + self.seq_len])
        peds = np.unique(seg[:, 1])

        buf = {
            k: []
            for k in (
                "obs_traj",
                "pred_traj",
                "obs_rel",
                "pred_rel",
                "obs_Me",
                "pred_Me",
                "obs_Me_rel",
                "pred_Me_rel",
                "nlp",
            )
        }
        cnt = 0

        for pid in peds:
            ps = seg[seg[:, 1] == pid]
            if len(ps) != self.seq_len:
                continue
            ps_t = np.transpose(ps[:, 2:])

            lon_deg = (ps_t[0, :] * 50.0 + 1800.0) / 10.0
            lat_deg = (ps_t[1, :] * 50.0) / 10.0
            if lon_deg.min() < _LON_VALID_MIN or lon_deg.max() > _LON_VALID_MAX:
                self._n_filtered_coord += 1
                continue
            if lat_deg.max() > _LAT_VALID_MAX:
                self._n_filtered_coord += 1
                continue

            ps_t_clipped = ps_t.copy()
            ps_t_clipped[0, :] = np.clip(ps_t[0, :], _LON_NORM_MIN, _LON_NORM_MAX)
            ps_t_clipped[1, :] = np.clip(ps_t[1, :], _LAT_NORM_MIN, _LAT_NORM_MAX)

            rel = np.zeros_like(ps_t_clipped)
            rel[:, 1:] = ps_t_clipped[:, 1:] - ps_t_clipped[:, :-1]

            buf["obs_traj"].append(torch.from_numpy(ps_t_clipped[:2, :obs_len]).float())
            buf["pred_traj"].append(torch.from_numpy(ps_t_clipped[:2, obs_len:]).float())
            buf["obs_rel"].append(torch.from_numpy(rel[:2, :obs_len]).float())
            buf["pred_rel"].append(torch.from_numpy(rel[:2, obs_len:]).float())
            buf["obs_Me"].append(torch.from_numpy(ps_t_clipped[2:, :obs_len]).float())
            buf["pred_Me"].append(torch.from_numpy(ps_t_clipped[2:, obs_len:]).float())
            buf["obs_Me_rel"].append(torch.from_numpy(rel[2:, :obs_len]).float())
            buf["pred_Me_rel"].append(torch.from_numpy(rel[2:, obs_len:]).float())
            buf["nlp"].append(self._poly_fit(ps_t_clipped, pred_len, threshold))
            cnt += 1

        if cnt < min_ped:
            return

        self.obs_traj_raw.extend(buf["obs_traj"])
        self.pred_traj_raw.extend(buf["pred_traj"])
        self.obs_rel_raw.extend(buf["obs_rel"])
        self.pred_rel_raw.extend(buf["pred_rel"])
        self.obs_Me_raw.extend(buf["obs_Me"])
        self.pred_Me_raw.extend(buf["pred_Me"])
        self.obs_Me_rel_raw.extend(buf["obs_Me_rel"])
        self.pred_Me_rel_raw.extend(buf["pred_Me_rel"])
        self.non_linear_ped.extend(buf["nlp"])
        self._num_peds_in_seq.append(cnt)
        self.tyID.append(
            {
                "old": [f_year, f_name, idx],
                "tydate": [add[i][0] for i in range(idx, idx + self.seq_len)],
            }
        )

    def _build_synthetic_lookup(self, csv_path: str) -> dict:
        try:
            import pandas as pd
        except ImportError:
            return {}
        if not os.path.exists(csv_path):
            return {}
        try:
            df = pd.read_csv(csv_path, dtype={"storm_name": str, "dt": str})
            lookup = {}
            for (yr, nm), grp in df.groupby(["year", "storm_name"]):
                lookup[(str(int(float(yr))), str(nm))] = grp.sort_values("step_i").reset_index(
                    drop=True
                )
            return lookup
        except Exception as e:
            logger.warning(f"Synthetic CSV load failed: {e}")
            return {}

    def _load_synthetic_sequences(self, min_ped, threshold, obs_len, pred_len, delim):
        num_new = []
        for (f_year, f_name), storm_df in self._synthetic_lookup.items():
            n_steps = len(storm_df)
            if n_steps < self.seq_len:
                continue
            n_seq = (n_steps - self.seq_len) // self.skip + 1
            dates_list = storm_df["dt"].tolist()
            for idx in range(0, n_seq * self.skip, self.skip):
                if idx + self.seq_len > n_steps:
                    break
                if self._process_synthetic_window(
                    storm_df, idx, f_year, f_name, dates_list, pred_len, threshold, obs_len
                ):
                    num_new.append(1)

        existing_end = self.seq_start_end[-1][1] if self.seq_start_end else 0
        new_cum = np.cumsum(num_new) + existing_end
        new_pairs = list(zip([existing_end] + new_cum[:-1].tolist(), new_cum.tolist()))
        self.seq_start_end.extend(new_pairs)

    def _process_synthetic_window(
        self, storm_df, idx, f_year, f_name, dates_list, pred_len, threshold, obs_len
    ):
        seg = storm_df.iloc[idx : idx + self.seq_len]
        lon_norm = seg["lon_norm"].values
        lat_norm = seg["lat_norm"].values
        pres_norm = seg["pres_norm"].values
        wnd_norm = seg["wnd_norm"].values

        lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
        lat_deg = (lat_norm * 50.0) / 10.0
        if lon_deg.min() < _LON_VALID_MIN or lon_deg.max() > _LON_VALID_MAX:
            return False
        if lat_deg.max() > _LAT_VALID_MAX:
            return False

        lon_norm = np.clip(lon_norm, _LON_NORM_MIN, _LON_NORM_MAX)
        lat_norm = np.clip(lat_norm, _LAT_NORM_MIN, _LAT_NORM_MAX)
        ps_t = np.stack([lon_norm, lat_norm, pres_norm, wnd_norm], axis=0)
        rel = np.zeros_like(ps_t)
        rel[:, 1:] = ps_t[:, 1:] - ps_t[:, :-1]

        self.obs_traj_raw.append(torch.from_numpy(ps_t[:2, :obs_len]).float())
        self.pred_traj_raw.append(torch.from_numpy(ps_t[:2, obs_len:]).float())
        self.obs_rel_raw.append(torch.from_numpy(rel[:2, :obs_len]).float())
        self.pred_rel_raw.append(torch.from_numpy(rel[:2, obs_len:]).float())
        self.obs_Me_raw.append(torch.from_numpy(ps_t[2:, :obs_len]).float())
        self.pred_Me_raw.append(torch.from_numpy(ps_t[2:, obs_len:]).float())
        self.obs_Me_rel_raw.append(torch.from_numpy(rel[2:, :obs_len]).float())
        self.pred_Me_rel_raw.append(torch.from_numpy(rel[2:, obs_len:]).float())
        self.non_linear_ped.append(self._poly_fit(ps_t, pred_len, threshold))

        tydate = [str(dates_list[i]) for i in range(idx, idx + self.seq_len)]
        self.tyID.append({"old": [f_year, f_name, idx], "tydate": tydate, "is_synthetic": True})
        return True

    def _read_file(self, path: str, delim: str) -> dict:
        data, add = [], []
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith(("#", "//", "-", "=")):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            try:
                frame_id = float(parts[0])
                lon_norm = float(parts[1])
                lat_norm = float(parts[2])
                pres_norm = float(parts[3])
                wnd_norm = float(parts[4])
                date = parts[5]
                name = parts[6]
                add.append([date, name])
                data.append([frame_id, 1.0, lon_norm, lat_norm, pres_norm, wnd_norm])
            except (ValueError, IndexError):
                continue
        return {
            "main": (
                np.asarray(data, dtype=np.float32) if data else np.zeros((0, 6), dtype=np.float32)
            ),
            "addition": add,
        }

    def _poly_fit(self, traj, tlen, threshold):
        t = np.linspace(0, tlen - 1, tlen)
        rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
        ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
        return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

    def _load_env_npy(self, year, ty_name, timestamp):
        folder = os.path.join(self.env_path, str(year), str(ty_name))
        if os.path.exists(folder):
            candidates = []
            for fname in [f"WP{year}{ty_name}_{timestamp}.npy", f"{timestamp}.npy"]:
                p = os.path.join(folder, fname)
                if os.path.exists(p):
                    candidates.append(p)
            if not candidates:
                try:
                    candidates = [
                        os.path.join(folder, f)
                        for f in sorted(os.listdir(folder))
                        if timestamp in f and f.endswith(".npy")
                    ]
                except Exception:
                    pass

            for p in candidates:
                try:
                    raw = np.load(p, allow_pickle=True).item()
                    if not isinstance(raw, dict):
                        continue
                    remapped = self._remap_npy_keys(raw)
                    return env_data_processing(remapped)
                except Exception as e:
                    logger.debug(f"env npy load error {p}: {e}")

        if self._csv_env_lookup:
            yr_str, ts_str = str(year), str(timestamp)
            for name_try in (
                str(ty_name),
                str(ty_name).lstrip("0") or "0",
                str(ty_name).zfill(4),
                str(ty_name).zfill(2),
            ):
                raw_dict = self._csv_env_lookup.get((yr_str, name_try, ts_str))
                if raw_dict is not None:
                    return env_data_processing(dict(raw_dict))

        return None

    @staticmethod
    def _remap_npy_keys(raw: dict) -> dict:
        remapped = dict(raw)
        has_old_gph = False
        has_old_n_suffix = any(k.endswith("_n") for k in remapped)

        for old_k, new_k in _NPY_KEY_REMAP.items():
            if old_k in remapped and new_k not in remapped:
                remapped[new_k] = remapped.pop(old_k)
                has_old_gph = True
        for old_k, new_k in _NPY_U500_KEY_REMAP.items():
            if old_k in remapped and new_k not in remapped:
                remapped[new_k] = remapped.pop(old_k)

        if has_old_gph:
            remapped["gph500_already_normed"] = True
        elif "gph500_mean" in remapped and not has_old_n_suffix:
            remapped["gph500_already_normed"] = True
        return remapped

    def img_read(self, year, ty_name, timestamp):
        return read_era5_step(self.era5_data_path, year, ty_name, timestamp)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        if self.num_seq == 0:
            raise IndexError("Empty dataset")
        s, e = self.seq_start_end[index]
        info = self.tyID[index]
        year = str(info["old"][0])
        tyname = str(info["old"][1])
        dates = info["tydate"]

        imgs = [self.img_read(year, tyname, ts) for ts in dates[: self.obs_len]]
        img_obs = torch.stack(imgs, dim=0)
        img_pred = torch.zeros(self.pred_len, ERA5_H, ERA5_W, ERA5_CH)

        obs_traj = torch.stack([self.obs_traj_raw[i] for i in range(s, e)])
        pred_traj = torch.stack([self.pred_traj_raw[i] for i in range(s, e)])
        obs_rel = torch.stack([self.obs_rel_raw[i] for i in range(s, e)])
        pred_rel = torch.stack([self.pred_rel_raw[i] for i in range(s, e)])
        obs_Me = torch.stack([self.obs_Me_raw[i] for i in range(s, e)])
        pred_Me = torch.stack([self.pred_Me_raw[i] for i in range(s, e)])
        obs_Me_rel = torch.stack([self.obs_Me_rel_raw[i] for i in range(s, e)])
        pred_Me_rel = torch.stack([self.pred_Me_rel_raw[i] for i in range(s, e)])

        n = e - s
        nlp = [self.non_linear_ped[i] for i in range(s, e)]
        mask = torch.ones(n, self.seq_len)

        obs_traj_np = obs_traj[0].numpy()
        obs_Me_np = obs_Me[0].numpy()

        cache_key = (year, tyname, tuple(dates[: self.obs_len]))
        if cache_key not in self.env_cache:
            self.env_cache[cache_key] = compute_env_features(
                self._load_env_npy, year, tyname, dates[: self.obs_len], obs_traj_np, obs_Me_np
            )
        env_out = self.env_cache[cache_key]

        return [
            obs_traj,
            pred_traj,
            obs_rel,
            pred_rel,
            nlp,
            mask,
            obs_Me,
            pred_Me,
            obs_Me_rel,
            pred_Me_rel,
            embed_time(dates[: self.obs_len]),
            embed_time(dates[self.obs_len :]),
            img_obs,
            img_pred,
            env_out,
            info,
        ]
