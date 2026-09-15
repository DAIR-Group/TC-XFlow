import shutil

import numpy as np

from train_fm.checkpoint import unwrap, save_checkpoint
from train_fm.evaluation import evaluate, evaluate_hard_val


def run_validation(
    model,
    val_loader,
    device,
    ema,
    args,
    ep,
    rel_ep,
    train_loss,
    best_score,
    patience_cnt,
    val_dpe_history,
    swa,
    opt,
    sched,
    scaler,
    best_ckpt,
    model_cfg,
    xai_batch,
):
    run_xai_this = rel_ep % 10 == 0
    r = evaluate(
        model,
        val_loader,
        device,
        tag=f"VAL ep{ep}",
        n_ensemble=args.n_ensemble,
        ema=ema,
        epoch_for_loss=ep,
        run_xai=run_xai_this,
        xai_batch=xai_batch,
    )

    val_dpe = r["DPE"]
    score = r["combined_score"]
    val_dpe_history.append(val_dpe)

    if len(val_dpe_history) >= 4:
        trend = float(np.mean(val_dpe_history[-2:])) - float(np.mean(val_dpe_history[-4:-2]))
        trend_s = (
            f"UP {trend:+.1f}km (worse)"
            if trend > 5
            else f"DOWN {trend:+.1f}km (better)" if trend < -5 else f"FLAT {trend:+.1f}km"
        )
    else:
        trend_s = "-"
    print(f"  train={train_loss:.6f}  val_DPE={val_dpe:.1f}  combined={score:.1f}  trend={trend_s}")

    if (
        not swa.active
        and ep >= args.swa_min_ep
        and swa.should_activate(val_dpe_history, args.swa_window, args.swa_threshold)
    ):
        swa.activate(model, opt, ep)

    stop = False
    if score < best_score:
        best_score = score
        patience_cnt = 0
        save_checkpoint(
            best_ckpt,
            ep,
            model,
            opt,
            sched,
            best_score,
            ema,
            scaler,
            extra={
                "val_dpe": r["DPE"],
                "val_ate": r["ATE"],
                "val_cte": r["CTE"],
                "patience_cnt": 0,
            },
            model_cfg=model_cfg,
        )
        print(
            f"  Best! score={best_score:.2f}"
            f"  Mean DPE={r['DPE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}"
        )
    else:
        if rel_ep >= args.min_ep and not swa.active:
            patience_cnt += args.val_freq
        print(
            f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})"
            f"{'  [SWA active -- patience frozen]' if swa.active else ''}"
        )
        if rel_ep >= args.min_ep and not swa.active and patience_cnt >= args.patience:
            print(f"  Early stop @ ep{ep}")
            stop = True

    return best_score, patience_cnt, val_dpe_history, stop


def run_hard_validation(
    model,
    val_loader,
    device,
    ema,
    args,
    ep,
    best_hard,
    hard_best_ckpt,
    model_cfg,
    opt,
    sched,
    scaler,
):
    r_h = evaluate_hard_val(
        model,
        val_loader,
        device,
        hard_threshold=args.hard_val_threshold,
        n_ensemble=args.n_ensemble,
        ema=ema,
        epoch_for_loss=ep,
    )
    print(
        f"  [HVAL] n={r_h['n_hard']}"
        f"  Mean DPE={r_h['DPE']:.1f} ATE={r_h['ATE']:.1f} CTE={r_h['CTE']:.1f}"
        f"  combined={r_h['combined_score']:.1f}"
    )
    if r_h["combined_score"] < best_hard and r_h["n_hard"] >= 10:
        best_hard = r_h["combined_score"]
        save_checkpoint(
            hard_best_ckpt,
            ep,
            model,
            opt,
            sched,
            best_hard,
            ema,
            scaler,
            extra={"hard_val_dpe": r_h["DPE"], "selection_criterion": "hard_val"},
            model_cfg=model_cfg,
        )
        print(f"  Hard-best! score={best_hard:.2f} Mean DPE={r_h['DPE']:.1f}")
    return best_hard


def try_swa_checkpoint(
    model,
    val_loader,
    device,
    swa,
    args,
    ep,
    best_score,
    patience_cnt,
    best_ckpt,
    swa_ckpt,
    model_cfg,
):
    backup = {
        k: v.detach().clone()
        for k, v in unwrap(model).state_dict().items()
        if v.dtype.is_floating_point
    }
    swa.apply_to_model(model)
    r_swa = evaluate(
        model,
        val_loader,
        device,
        tag=f"SWA ep{ep}",
        n_ensemble=args.n_ensemble,
        ema=None,
        epoch_for_loss=ep,
    )
    swa.restore_from_backup(model, backup)

    swa_score = r_swa["combined_score"]
    print(f"  [SWA] score={swa_score:.2f} ({swa.n_updates} updates) vs best={best_score:.2f}")
    if swa_score < best_score:
        best_score = swa_score
        patience_cnt = 0
        swa.save_avg_state(
            swa_ckpt,
            ep,
            best_score,
            extra={"val_dpe": r_swa["DPE"], "val_ate": r_swa["ATE"], "val_cte": r_swa["CTE"]},
            model_cfg=model_cfg,
        )
        shutil.copy(swa_ckpt, best_ckpt)
        print(f"  SWA best! score={best_score:.2f} Mean DPE={r_swa['DPE']:.1f}")
    return best_score, patience_cnt
