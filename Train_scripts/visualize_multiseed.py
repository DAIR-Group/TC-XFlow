from __future__ import annotations

import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from Model.Data.loader import data_loader

from vis.geo_utils import set_seed
from vis.dataset_lookup import resolve_date, find_target, list_available
from vis.inference import load_tcxflow_checkpoint, run_tcxflow_inference
from vis.plotting import infer_seed_label, plot_multi_seed_comparison
from vis.args import get_args


def visualize_multi_seed(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_name = args.tc_name.strip().upper()
    t_date, _ = resolve_date(args.tc_date)

    print("=" * 65)
    print(f"  TC-XFlow Visualize — Multi-seed comparison  |  {t_name}  @  {t_date}")
    print("=" * 65 + "\n")

    if not args.seed_checkpoints:
        print(
            "  ERROR: --seed_checkpoints cần ít nhất 1 checkpoint (khuyến nghị >=2 để so sánh có ý nghĩa)"
        )
        return

    dset, _ = data_loader(
        args,
        {"root": args.TC_data_path, "type": args.dset_type},
        test=True,
        test_year=args.test_year,
    )
    print(f"  Dataset: {len(dset)} samples\n")

    target, _, actual_date = find_target(dset, t_name, t_date, args.obs_len)
    if target is None:
        print(f"  '{t_name} @ {t_date}' not found.")
        list_available(dset, t_name, args.obs_len)
        return
    t_date = actual_date
    print(f"  Found: {t_name} @ {t_date}\n")

    preds_by_seed, errors_by_seed = {}, {}
    winds_pred_by_seed, wind_gt = {}, None
    obs_deg = gt_deg = None

    for idx, ckpt in enumerate(args.seed_checkpoints):
        seed_label = infer_seed_label(ckpt, idx)
        print(f"  Loading seed={seed_label}: {ckpt}")
        model = load_tcxflow_checkpoint(ckpt, device, obs_len=args.obs_len, pred_len=args.pred_len)
        od, gd, pd_, _ens, err, wpred, wgt = run_tcxflow_inference(
            model, target, device, ode_steps=args.ode_steps, num_ensemble=args.num_ensemble
        )
        obs_deg, gt_deg = od, gd
        preds_by_seed[seed_label] = pd_
        errors_by_seed[seed_label] = err
        winds_pred_by_seed[seed_label] = wpred
        if wind_gt is None:
            wind_gt = wgt
        print(f"    seed={seed_label}: Mean DPE={err.mean():.1f}km")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"track_multiseed_fm_{t_name}_{t_date}.png")
    plot_multi_seed_comparison(
        obs_deg,
        gt_deg,
        preds_by_seed,
        errors_by_seed,
        f"{t_name} (TC-XFlow)",
        out_path,
        winds_pred_by_seed=winds_pred_by_seed,
        wind_gt=wind_gt,
    )


if __name__ == "__main__":
    visualize_multi_seed(get_args())
