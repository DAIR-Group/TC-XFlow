
from collections import defaultdict
from typing import Dict

import numpy as np
import torch

from Model.Main_model.loss import (
    _norm_to_deg,
    _haversine_deg,
    hard_score_from_obs,
)

from .checkpoint import move
from .metrics import HORIZON_STEPS, ate_cte


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def evaluate_hard_val(
    model,
    val_loader,
    device,
    hard_threshold: float = 0.35,
    n_ensemble: int = 20,
    ema=None,
    epoch_for_loss: int = 9999,
):
   
    backup = ema.apply_to(model) if ema is not None else None

    model.eval()
    all_dpe, all_ate, all_cte = [], [], []
    n_hard = 0

    for batch in val_loader:
        bl = move(list(batch), device)
        B = bl[0].shape[1]
        hard_score = hard_score_from_obs(bl[0][:, :, :2])
        hard_mask = hard_score > hard_threshold
        if hard_mask.sum() == 0:
            continue
        hard_idx = hard_mask.nonzero(as_tuple=True)[0]

        bl_hard = list(bl)
        for i, item in enumerate(bl_hard):
            if torch.is_tensor(item):
                if item.dim() >= 2 and item.shape[1] == B:
                    bl_hard[i] = item[:, hard_idx, ...]
                elif item.dim() >= 1 and item.shape[0] == B:
                    bl_hard[i] = item[hard_idx, ...]

        try:
            pred, _, _ = model.sample(bl_hard, num_ensemble=n_ensemble, use_curvature_score=True)
        except Exception as e:
            print(f"  hard val error: {e}")
            continue

        gt = bl_hard[1]
        T = min(pred.shape[0], gt.shape[0])
        pred_deg = _norm_to_deg(pred[:T])
        gt_deg = _norm_to_deg(gt[:T])
        dist = _haversine_deg(pred_deg, gt_deg)
        along, cross = ate_cte(pred_deg, gt_deg)

        all_dpe.extend(dist.mean(0).tolist())
        if along.shape[0] > 0:
            all_ate.extend(along.abs().mean(0).tolist())
            all_cte.extend(cross.abs().mean(0).tolist())
        n_hard += len(hard_idx)

    if backup is not None:
        ema.restore(model, backup)

    dpe, ate, cte = _mean(all_dpe), _mean(all_ate), _mean(all_cte)
    return {
        "DPE": dpe,
        "ATE": ate,
        "CTE": cte,
        "n_hard": n_hard,
        "combined_score": 0.6 * dpe + 0.2 * ate + 0.2 * cte,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    tag: str = "",
    n_ensemble: int = 20,
    ema=None,
    use_tta: bool = False,
    n_tta: int = 5,
    epoch_for_loss: int = 9999,
    run_xai: bool = False,
    xai_batch=None,
) -> Dict:
   
  
    backup = ema.apply_to(model) if ema is not None else None

    model.eval()
    all_dpe, all_ate, all_cte = [], [], []
    step_dist = defaultdict(list)
    sum_loss = sum_cfm = sum_head = 0.0
    sum_n = 0

    for batch in loader:
        bl = move(list(batch), device)
        gt = bl[1]
        B = bl[0].shape[1]

        try:
            bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
            if torch.isfinite(bd["total"]):
                sum_loss += bd["total"].item() * B
                sum_cfm += bd["l_cfm"] * B
                sum_head += bd["l_heading"] * B
                sum_n += B
        except Exception:
            pass

        pred = _predict(model, bl, n_ensemble, use_tta, n_tta)
        if pred is None:
            continue

        T = min(pred.shape[0], gt.shape[0])
        pred_deg = _norm_to_deg(pred[:T])
        gt_deg = _norm_to_deg(gt[:T])
        dist = _haversine_deg(pred_deg, gt_deg)
        along, cross = ate_cte(pred_deg, gt_deg)

        all_dpe.extend(dist.mean(0).tolist())
        if along.shape[0] > 0:
            all_ate.extend(along.abs().mean(0).tolist())
            all_cte.extend(cross.abs().mean(0).tolist())
        for h, step in HORIZON_STEPS.items():
            if step < T:
                step_dist[h].extend(dist[step].tolist())

    if backup is not None:
        ema.restore(model, backup)

    val_loss = sum_loss / max(sum_n, 1)
    dpe, ate, cte = _mean(all_dpe), _mean(all_ate), _mean(all_cte)

    result = {
        "DPE": dpe,
        "ATE": ate,
        "CTE": cte,
        "n": len(all_dpe),
        "val_loss": val_loss,
        "val_cfm_loss": sum_cfm / max(sum_n, 1),
        "val_head_loss": sum_head / max(sum_n, 1),
        "val_mom_loss": 0.0,
    }
    for h in HORIZON_STEPS:
        result[f"{h}h"] = _mean(step_dist[h])
    result["combined_score"] = (
        0.6 * dpe + 0.2 * ate + 0.2 * cte if all(np.isfinite(x) for x in [dpe, ate, cte]) else dpe
    )

    _print_summary(result, tag, use_tta)

    if run_xai and xai_batch is not None:
        _print_xai_summary(model, xai_batch, result)

    print(f"  {'='*72}\n")
    return result


def _predict(model, bl, n_ensemble, use_tta, n_tta):
   
    if not use_tta:
        try:
            pred, _, _ = model.sample(bl, num_ensemble=n_ensemble, use_curvature_score=True)
            return pred
        except Exception as e:
            print(f"  sample error: {e}")
            return None

    obs = bl[0]
    anchor = obs[-1:, :, :2].detach()
    scales = [0.875, 0.9375, 1.0, 1.0625, 1.125][:n_tta]
    preds, weights = [], []
    for scale in scales:
        obs_scaled = obs.clone()
        obs_scaled[..., :2] = anchor + (obs[..., :2] - anchor) * scale
        bl_scaled = list(bl)
        bl_scaled[0] = obs_scaled
        try:
            p, _, _ = model.sample(bl_scaled, num_ensemble=n_ensemble, use_curvature_score=True)
            preds.append(p)
            weights.append(2.0 if abs(scale - 1.0) < 1e-6 else 1.0)
        except Exception:
            continue

    if not preds:
        return None
    total_weight = sum(weights)
    return sum(w / total_weight * p for w, p in zip(weights, preds))


def _print_summary(result, tag, use_tta):
    def v(key):
        return result.get(key, float("nan"))

    tta_str = " [TTA]" if use_tta else ""
    print(f"\n  {'='*72}")
    print(f"  [{tag}]{tta_str}  n={result['n']}")
    print(
        f"  Val Loss : {result['val_loss']:.6f}  cfm={result['val_cfm_loss']:.6f}  "
        f"head4s={result['val_head_loss']:.6f}  [mom=DISABLED]"
    )
    print(f"  Mean DPE={v('DPE'):7.1f}km  ATE={v('ATE'):7.1f}km  CTE={v('CTE'):7.1f}km")
    print(f"  Combined = {v('combined_score'):.1f}")
    print(
        f"  12h={v('12h'):6.1f}  24h={v('24h'):6.1f}  48h={v('48h'):6.1f}  72h={v('72h'):6.1f} km"
    )


def _print_xai_summary(model, xai_batch, result):
    from .checkpoint import unwrap

    try:
        _, _, _, xai = unwrap(model).sample(xai_batch, return_xai=True, use_curvature_score=True)
    except Exception as e:
        print(f"  XAI error: {e}")
        return

    print(f"  {'-'*60}")
    print("  XAI Summary")
    print(f"  {'-'*60}")

    print(
        f"  [XAI-4] Uncertainty:"
        f" 12h={xai['mean_12h_std']:.1f}km"
        f"  72h={xai['mean_72h_std']:.1f}km"
        f"  ratio={float(xai['uncertainty_ratio'].mean()):.2f}×"
        f"  high_uncert={xai['high_uncertainty'].sum().item()}"
    )

    hard_components = xai.get("hard_components", {})
    if hard_components:
        print(
            f"  [XAI-2] HardScore:"
            f" curv={float(hard_components['curvature'].mean()):.3f}"
            f"  spd_var={float(hard_components['speed_var'].mean()):.3f}"
            f"  dir_chg={float(hard_components['dir_change'].mean()):.3f}"
            f"  obs_spd_n={float(hard_components.get('obs_speed_norm', torch.zeros(1)).mean()):.3f}"
        )

    physics_components = xai.get("physics_components", {})
    if physics_components:
        print(
            f"  [XAI-3] Physics:"
            f" speed={float(physics_components['speed'].mean()):.3f}"
            f"  smooth={float(physics_components['smooth'].mean()):.3f}"
            f"  heading={float(physics_components['heading'].mean()):.3f}"
        )

    speed_cmp = xai.get("speed_comparison", {})
    if speed_cmp:
        ratio = speed_cmp.get("speed_ratio", 1.0)
        flag = "OVER" if ratio > 1.15 else "UNDER" if ratio < 0.85 else "OK"
        n_over = speed_cmp.get("over_predict", torch.zeros(1)).sum().item()
        n_under = speed_cmp.get("under_predict", torch.zeros(1)).sum().item()
        print(
            f"  [XAI-5] Speed (post-calibration):"
            f" obs={speed_cmp['obs_speed_mean']:.1f}km/h"
            f"  pred={speed_cmp['pred_speed_mean']:.1f}km/h"
            f"  ratio={ratio:.2f} {flag}"
            f"  (over:{int(n_over)} under:{int(n_under)})"
        )

    heading_dev = xai.get("heading_deviation_deg")
    if heading_dev is not None and heading_dev.shape[0] >= 1:
        hd_mean = heading_dev.mean(1)
        print(
            f"  [XAI-6] Heading deviation:"
            f" 12h={hd_mean[0].item():.1f}°"
            f"  24h={hd_mean[min(2, len(hd_mean)-1)].item():.1f}°"
            f"  72h={hd_mean[min(10, len(hd_mean)-1)].item():.1f}°"
        )

    ate_cte_decomp = xai.get("ate_cte_decomp", {})
    if ate_cte_decomp:
        print(
            f"  [XAI-7] Error:"
            f" ATE={ate_cte_decomp['ate_abs_mean']:.1f}km"
            f"  CTE={ate_cte_decomp['cte_abs_mean']:.1f}km"
            f"  ratio={ate_cte_decomp['ate_abs_mean']/(ate_cte_decomp['cte_abs_mean']+1e-3):.2f}"
        )

    speed_per_horizon = xai.get("speed_per_horizon", {})
    if speed_per_horizon and "pred_kmh" in speed_per_horizon:
        ratios = speed_per_horizon["ratio"]
        horizons = [(0, "12h"), (2, "24h"), (6, "48h"), (10, "72h")]
        parts = []
        for idx, label in horizons:
            if idx < len(ratios):
                flag = "FAIL" if ratios[idx] > 1.3 or ratios[idx] < 0.7 else "OK"
                parts.append(f"{label}:r={ratios[idx]:.2f}{flag}")
        print(f"  [XAI-8] Speed/horizon: {' | '.join(parts)}")

    storm_categories = xai.get("storm_categories", {})
    if storm_categories:
        print(
            f"  [XAI-9] Storms:"
            f" slow={storm_categories.get('n_slow', 0)}"
            f"  med={storm_categories.get('n_medium', 0)}"
            f"  fast={storm_categories.get('n_fast', 0)}"
            f"  spd={storm_categories.get('speed_mean', 0):.1f}"
            f"±{storm_categories.get('speed_std', 0):.1f}km/h"
        )

    learned = xai.get("learned_params", {})
    if learned:
        speed_corr_str = ",".join(f"{v:.2f}" for v in learned.get("speed_correction", [])[:4])
        reg_ema_full = learned.get("reg_dist_ema_km_per_horizon", [])
        reg_ema_near = ",".join(f"{v:.1f}" for v in reg_ema_full[:4])
        reg_ema_far = (
            ",".join(f"{v:.1f}" for v in reg_ema_full[7:12]) if len(reg_ema_full) >= 12 else "?"
        )
        heading_ema_full = learned.get("heading_err_ema_per_horizon", [])
        heading_ema_str = ",".join(f"{v:.3f}" for v in heading_ema_full[:4])
        hard_weights_str = ",".join(f"{v:.3f}" for v in learned.get("hard_score_weights", []))
        sigma_inf = learned.get("sigma_inf", float("nan"))
        log_sigma_reg = learned.get("log_sigma_reg", float("nan"))
        log_sigma_heading = learned.get("log_sigma_heading", float("nan"))
        log_sigma_calib = learned.get("log_sigma_calib", float("nan"))
        eff_lambda_reg = learned.get("eff_lambda_reg", float("nan"))
        eff_lambda_heading = learned.get("eff_lambda_heading", float("nan"))
        eff_lambda_calib = learned.get("eff_lambda_calib", float("nan"))
        print(
            f"  [LEARN] speed_corr(12h-24h)=[{speed_corr_str}]  reg_ema_km(6h-24h)=[{reg_ema_near}]"
        )
        print(
            f"  [LEARN] reg_ema_km(48h-72h,idx7-11)=[{reg_ema_far}]"
            f"  heading_ema(6h-24h)=[{heading_ema_str}]"
        )
        print(
            f"  [LEARN] hard_w(curv,spdvar,dirchg,obsspd)=[{hard_weights_str}]"
            f"  sigma_inf={sigma_inf:.4f}"
        )
        print(
            f"  [LEARN] log_sigma: reg={log_sigma_reg:.3f}  heading={log_sigma_heading:.3f}"
            f"  calib={log_sigma_calib:.3f}"
            f"  |  eff_lambda: reg={eff_lambda_reg:.3f}  heading={eff_lambda_heading:.3f}"
            f"  calib={eff_lambda_calib:.3f}"
        )

    print(f"  {'-'*60}")
    result["xai"] = xai
