from datetime import datetime, timedelta

import torch


def detect_pred_len(ckpt_path):
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg = ck.get("model_cfg")
        if model_cfg and "pred_len" in model_cfg:
            return model_cfg["pred_len"]

        sd = ck.get("model", ck.get("model_state_dict", ck.get("model_state", ck)))
        for key in ["velocity.pos_emb", "net.pos_emb", "pos_emb"]:
            if key in sd:
                return sd[key].shape[1]
        for key in ["velocity.step_emb.weight", "step_emb.weight"]:
            if key in sd:
                return sd[key].shape[0]
    except Exception as e:
        print(f"  [WARN] detect_pred_len failed ({e}); defaulting to 12")
    return 12


def snap_to_6h(date_str: str) -> str:
    s = str(date_str).strip()[:10]
    dt = datetime.strptime(s, "%Y%m%d%H")
    dt = dt.replace(hour=(dt.hour // 6) * 6, minute=0, second=0)
    return dt.strftime("%Y%m%d%H")


def resolve_date(raw_date: str) -> tuple[str, bool]:
    original = str(raw_date).strip()[:10]
    snapped = snap_to_6h(original)
    was_snapped = snapped != original
    if was_snapped:
        print(f"  [SNAP] {original} to {snapped}  ")
    return snapped, was_snapped


def _search_one_date(dset, t_name: str, t_date: str, obs_len: int):
    best_item = None
    best_idx = None
    best_pri = float("inf")

    for i in range(len(dset)):
        item = dset[i]
        info = item[-1]
        if t_name not in str(info["old"][1]).strip().upper():
            continue
        for idx, td in enumerate(info["tydate"]):
            if str(td).strip() != t_date:
                continue
            if idx < obs_len:
                continue
            pri = 0 if idx == obs_len else (idx - obs_len + 1)
            if pri < best_pri:
                best_item, best_idx, best_pri = item, idx, pri

    return best_item, best_idx


def find_target(dset, t_name: str, t_date: str, obs_len: int, max_forward_steps: int = 20):
    dt = datetime.strptime(t_date, "%Y%m%d%H")

    for step in range(max_forward_steps + 1):
        candidate = dt.strftime("%Y%m%d%H")
        item, idx = _search_one_date(dset, t_name, candidate, obs_len)
        if item is not None:
            if step > 0:
                print(
                    f"  [AUTO-FORWARD] {t_date} no data  " f" forward to {candidate}  (+{step*6}h)"
                )
            return item, idx, candidate
        dt = dt + timedelta(hours=6)

    return None, None, None


def list_available(dset, t_name: str, obs_len: int, limit: int = 30):
    shown = 0
    seen = set()
    for i in range(len(dset)):
        info = dset[i][-1]
        name = str(info["old"][1]).strip().upper()
        if t_name not in name:
            continue
        td = str(info["tydate"][obs_len]).strip()
        if td in seen:
            continue
        seen.add(td)
        print(f"    {name:<15s}  @  {td}")
        shown += 1
        if shown >= limit:
            break

    if shown == 0:
        print(f"  (TC '{t_name}' not found in the dataset)")
        print("  Some available TCs:")
        seen_names: set[str] = set()
        for i in range(len(dset)):
            info = dset[i][-1]
            n = str(info["old"][1]).strip().upper()
            if n in seen_names:
                continue
            seen_names.add(n)
            print(f"    {n}")
            if len(seen_names) >= 15:
                break
