import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch

from Model.Data.loader import data_loader
from Model.Main_model.loss import _norm_to_deg, _haversine_deg

from train_fm.checkpoint import unwrap, move
from train_fm.evaluation import evaluate
from train_fm.metrics import ate_cte, HORIZON_STEPS


def run_final_test(args, best_ckpt, model, ema, xai_batch, device):
    print("\n  Loading best checkpoint ")
    if not os.path.exists(best_ckpt):
        print("  No checkpoint found.")
        return

    ck = torch.load(best_ckpt, map_location=device)
    is_swa = ck.get("is_swa", False)
    unwrap(model).load_state_dict(ck["model"], strict=False)
    if not is_swa and ema and ck.get("ema"):
        for k, v in ck["ema"].items():
            if k in ema.shadow:
                ema.shadow[k].copy_(v.to(device))
    print(f"  Loaded ep{ck.get('epoch', '?')} (is_swa={is_swa})")

    try:
        _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
        print(f"  Test: {len(test_loader)} batches")
    except Exception:
        print("  No test set -> using val")
        _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)

    r_test = evaluate(
        model,
        test_loader,
        device,
        tag="TEST (best ckpt)",
        n_ensemble=args.n_ensemble,
        ema=None if is_swa else ema,
        run_xai=True,
        xai_batch=xai_batch,
    )

    if args.tta_test:
        r_tta = evaluate(
            model,
            test_loader,
            device,
            tag="TEST+TTA",
            n_ensemble=args.n_ensemble,
            ema=None if is_swa else ema,
            use_tta=True,
            n_tta=args.n_tta,
        )
        print(
            f"\n  TTA: Mean DPE {r_test['DPE']:.1f}->{r_tta['DPE']:.1f}  "
            f"ATE {r_test['ATE']:.1f}->{r_tta['ATE']:.1f}  "
            f"CTE {r_test['CTE']:.1f}->{r_tta['CTE']:.1f}"
        )

    if args.multiscale_test:
        run_multiscale_test(model, test_loader, device, ema, is_swa)

    val_dpe_b = ck.get("val_dpe", ck.get("val_ade", float("nan")))
    val_ate_b = ck.get("val_ate", float("nan"))
    val_cte_b = ck.get("val_cte", float("nan"))
    print("\n" + "=" * 72)
    print("  FINAL RESULTS")
    print("=" * 72)
    print(f"  {'Metric':<10} {'Val':>10} {'Test':>10} {'Gap':>10}")
    print("  " + "-" * 44)
    for metric_name, val_v, test_v in [
        ("Mean DPE", val_dpe_b, r_test["DPE"]),
        ("ATE", val_ate_b, r_test["ATE"]),
        ("CTE", val_cte_b, r_test["CTE"]),
    ]:
        gap = test_v - val_v if (np.isfinite(test_v) and np.isfinite(val_v)) else float("nan")
        print(f"  {metric_name:<10} {val_v:>10.2f} {test_v:>10.2f} {gap:>+10.2f}")
    print("=" * 72)

    auto_eval(args, best_ckpt, device)


def run_multiscale_test(model, test_loader, device, ema, is_swa):
    raw_m = unwrap(model)
    if not hasattr(raw_m, "sample_multiscale"):
        return

    print("\n  Multi-scale sigma test...")
    ms_dpes, ms_ates, ms_ctes = [], [], []
    ms_steps = defaultdict(list)
    raw_m.eval()
    backup = ema.apply_to(model) if (ema and not is_swa) else None
    with torch.no_grad():
        for batch in test_loader:
            bl = move(list(batch), device)
            try:
                pred, _, _ = raw_m.sample_multiscale(bl)
            except Exception:
                continue
            gt = bl[1]
            T = min(pred.shape[0], gt.shape[0])
            pred_deg = _norm_to_deg(pred[:T])
            gt_deg = _norm_to_deg(gt[:T])
            dist = _haversine_deg(pred_deg, gt_deg)
            along, cross = ate_cte(pred_deg, gt_deg)
            ms_dpes.extend(dist.mean(0).tolist())
            if along.shape[0] > 0:
                ms_ates.extend(along.abs().mean(0).tolist())
                ms_ctes.extend(cross.abs().mean(0).tolist())
            for h, step in HORIZON_STEPS.items():
                if step < T:
                    ms_steps[h].extend(dist[step].tolist())
    if backup:
        ema.restore(model, backup)

    def mean_or_nan(lst):
        return float(np.mean(lst)) if lst else float("nan")

    print(
        f"  Multi-scale: Mean DPE={mean_or_nan(ms_dpes):.1f} "
        f"ATE={mean_or_nan(ms_ates):.1f} CTE={mean_or_nan(ms_ctes):.1f}"
    )


def auto_eval(args, best_ckpt: str, device):
    print("\n" + "=" * 72)
    print("  AUTO EVALUATE (mean DPE / ATE / CTE on test set)")
    print("=" * 72)

    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates_eval = [
        os.path.join(project_root, "evaluate_full.py"),
        os.path.join(os.getcwd(), "evaluate_full.py"),
        os.path.join(script_dir, "evaluate_full.py"),
    ]
    eval_script = next((p for p in candidates_eval if os.path.exists(p)), None)

    eval_json = None
    if eval_script is None:
        print(f"    Đặt file tại: {candidates_eval[0]}")
    else:
        print(f"  RUN: evaluate_full.py ({eval_script})")
        cmd_eval = [
            sys.executable,
            eval_script,
            "--checkpoint",
            best_ckpt,
            "--dataset_root",
            args.dataset_root,
            "--split",
            "test",
            "--output_dir",
            eval_dir,
            "--n_ensemble",
            str(args.n_ensemble),
            "--no_crps",
            "--gpu",
            str(args.gpu_num),
        ]
        try:
            result = subprocess.run(cmd_eval, capture_output=False, timeout=1800)
            if result.returncode == 0:
                print(f"  OK: evaluate_full done -> {eval_dir}/")
            else:
                print(f"  FAIL: evaluate_full failed (code {result.returncode})")
        except subprocess.TimeoutExpired:
            print("  WARN: evaluate_full timeout (30min)")
        except Exception as e:
            print(f"  WARN: evaluate_full error: {e}")

        if os.path.exists(eval_dir):
            jsons = sorted(
                os.path.join(eval_dir, f)
                for f in os.listdir(eval_dir)
                if "test" in f and f.endswith(".json")
            )
            if jsons:
                eval_json = jsons[-1]

    summary_path = os.path.join(args.output_dir, "auto_eval_summary.json")
    try:
        summary = {
            "checkpoint": best_ckpt,
            "eval_dir": eval_dir,
            "eval_json": eval_json,
            "seed": getattr(args, "seed", 42),
            "ablation_name": getattr(args, "ablation_name", ""),
        }
        if eval_json and os.path.exists(eval_json):
            with open(eval_json) as f:
                ev = json.load(f)
            summary["test_DPE"] = ev.get("DPE", ev.get("ADE"))
            summary["test_ATE"] = ev.get("ATE")
            summary["test_CTE"] = ev.get("CTE")
            summary["test_RMSE"] = ev.get("RMSE")
            summary["crps_mean"] = ev.get("crps", {}).get("mean")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary -> {summary_path}")
        if summary.get("test_DPE"):
            print(
                f"  Mean DPE={summary['test_DPE']:.2f}  "
                f"ATE={summary['test_ATE']:.2f}  "
                f"CTE={summary['test_CTE']:.2f}"
            )
    except Exception as e:
        print(f"  Summary save failed: {e}")

    print("=" * 72)
