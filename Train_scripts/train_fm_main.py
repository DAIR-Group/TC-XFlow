from __future__ import annotations
import json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import time

import numpy as np
import torch
from torch.amp import GradScaler

from Model.Data.loader import data_loader
from Model.main_model import Main_model

from train_fm.checkpoint import unwrap, move, save_checkpoint, resume_from_checkpoint
from train_fm.schedulers import build_optimizer, TwoGroupScheduler, SWAHandler
from train_fm.args import get_args
from train_fm.loss_ablation import apply_ablation_patch
from train_fm.epoch_loop import train_one_epoch
from train_fm.validation import run_validation, run_hard_validation, try_swa_checkpoint
from train_fm.test_eval import run_final_test


def main(args):
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.ablation_name:
        args.output_dir = f"{args.output_dir}_{args.ablation_name}"
    if args.seed != 42:
        args.output_dir = f"{args.output_dir}_seed{args.seed}"

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(
            f"  GPU: {torch.cuda.get_device_name(0)}  "
            f"(CUDA {torch.version.cuda}, {torch.cuda.device_count()} visible)"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    best_ckpt = os.path.join(args.output_dir, "best_model.pth")
    hard_best_ckpt = os.path.join(args.output_dir, "hard_best_model.pth")
    swa_ckpt = os.path.join(args.output_dir, "swa_model.pth")
    last_ckpt = os.path.join(args.output_dir, "last_model.pth")

    wall_start = time.time()

    print("=" * 72)
    print("  TC-XFlow")
    print("=" * 72)

    print("\n  Loading data...")
    trd, trl = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False, for_training=True
    )
    vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
    print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
    print(f"  val:   {len(vd)} ({len(val_loader)} batches)")

    try:
        xai_batch = move(list(next(iter(val_loader))), device)
    except Exception:
        xai_batch = None

    model_cfg = dict(
        pred_len=args.pred_len,
        obs_len=args.obs_len,
        unet_in_ch=args.unet_in_ch,
        d_cond=args.d_cond,
        d_model=args.d_model,
        nhead=args.nhead,
        num_dec_layers=args.num_dec_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_decay_end=args.sigma_decay_end,
        lambda_reg=args.lambda_reg,
        lambda_heading=args.lambda_heading,
        lambda_momentum=0.0,
        lambda_hard_reg=(0.0 if args.disable_hard_reg else args.lambda_hard_reg),
        log_sigma_reg_min_clamp=args.log_sigma_reg_min_clamp,
        enable_horizon_nll=not args.disable_horizon_nll,
        use_ot=args.use_ot,
        ot_epsilon=args.ot_epsilon,
        use_ema=args.use_ema,
        n_ensemble=args.n_ensemble,
        n_inference_steps=args.n_inference_steps,
        sigma_inference=args.sigma_inference,
        use_curvature_score_train=args.use_curvature_score_train,
    )
    model = Main_model(**model_cfg).to(device)

    model.init_ema()
    ema = getattr(unwrap(model), "_ema", None)
    raw = unwrap(model)
    n_enc = sum(p.numel() for p in raw.encoder.parameters())
    n_vel = sum(p.numel() for p in raw.velocity.parameters())
    encoder_ids = {id(p) for p in raw.encoder.parameters()}
    velocity_ids = {id(p) for p in raw.velocity.parameters()}
    n_extra = sum(
        p.numel()
        for p in raw.parameters()
        if id(p) not in encoder_ids and id(p) not in velocity_ids
    )
    n_total = n_enc + n_vel + n_extra
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(
        f"\n  Encoder: {n_enc:,}  VelocityTrans: {n_vel:,}  "
        f"LearnableExtra: {n_extra:,}  Total: {n_total:,}  Mem: {mem_mb:.1f}MB"
    )

    footprint_info = {
        "n_encoder": n_enc,
        "n_velocity": n_vel,
        "n_extra": n_extra,
        "n_total": n_total,
        "mem_mb": mem_mb,
        "seed": args.seed,
        "ablation_name": args.ablation_name or "full",
        "disable_l_heading": args.disable_l_heading,
        "disable_l_calib": args.disable_l_calib,
        "disable_l_reg": args.disable_l_reg,
        "disable_aug_c": args.disable_aug_c,
        "disable_hard_reg": args.disable_hard_reg,
        "lambda_hard_reg": args.lambda_hard_reg,
        "disable_horizon_nll": args.disable_horizon_nll,
        "use_ot": args.use_ot,
    }

    if args.disable_horizon_nll or not args.use_ot:
        print(
            f"  [ABLATION] Model-level: "
            f"{'no_horizon_nll(raw dist, no log_b_horizon) ' if args.disable_horizon_nll else ''}"
            f"{'no_OT(random x0/x1 pairing) ' if not args.use_ot else ''}"
        )

    if any(
        [
            args.disable_l_heading,
            args.disable_l_calib,
            args.disable_l_reg,
            args.disable_aug_c,
            args.disable_learned_weights,
            args.disable_hard_reg,
        ]
    ):
        apply_ablation_patch(raw, args)

    opt = build_optimizer(
        model,
        lr_velocity=args.lr,
        lr_encoder=0.0,
        weight_decay=args.weight_decay,
        lr_logits_scale=args.lr_logits_scale,
        lr_extra_scale=args.lr_extra_scale,
    )
    scaler = GradScaler("cuda", enabled=args.use_amp)
    sched = TwoGroupScheduler(
        opt=opt,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs,
        lr_vel=args.lr,
        lr_vel_min=args.lr_min,
        freeze_end_ep=args.freeze_encoder_epochs,
        lr_enc_peak=args.lr_enc_peak,
        encoder_warmup_epochs=args.encoder_warmup_epochs,
    )
    print(
        f"\n  LR vel: {args.lr:.0e} -> {args.lr_min:.0e}  "
        f"LR enc: 0 ({args.freeze_encoder_epochs}ep) -> {args.lr_enc_peak:.0e}"
    )

    swa = SWAHandler(swa_lr=args.swa_lr)

    start_ep = 0
    best_score = float("inf")
    best_hard = float("inf")
    patience_cnt = 0
    val_dpe_history = []

    if args.resume and os.path.exists(args.resume):
        start_ep, best_score, patience_cnt = resume_from_checkpoint(
            args.resume, model, opt, sched, scaler, ema, device
        )

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: ok")
    except Exception:
        pass

    nstep = len(trl)
    print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")
    print()

    for ep in range(start_ep, start_ep + args.num_epochs):
        rel_ep = ep - start_ep
        freeze = rel_ep < args.freeze_encoder_epochs
        for p in unwrap(model).encoder.parameters():
            p.requires_grad_(not freeze)

        if rel_ep == 0 and freeze:
            print(f"  *** Ep{ep}: encoder frozen ***")
        if rel_ep == args.freeze_encoder_epochs:
            print(f"\n  *** Ep{ep}: encoder unfrozen ***")

        train_loss = train_one_epoch(model, trl, opt, scaler, swa, args, ep, freeze, nstep, device)

        if not swa.active:
            sched.step()

        save_checkpoint(
            last_ckpt, ep, model, opt, sched, best_score, ema, scaler, model_cfg=model_cfg
        )
        if ep % 5 == 0:
            ep_ckpt = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
            save_checkpoint(
                ep_ckpt, ep, model, opt, sched, best_score, ema, scaler, model_cfg=model_cfg
            )
            print(f"  Saved: {ep_ckpt}")

        if rel_ep % args.val_freq == 0:
            best_score, patience_cnt, val_dpe_history, stop = run_validation(
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
            )
            if stop:
                break

        if rel_ep % args.hard_val_freq == 0 and rel_ep >= args.min_ep:
            best_hard = run_hard_validation(
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
            )

        if swa.active and rel_ep % args.val_freq == 0 and swa.n_updates >= 10:
            best_score, patience_cnt = try_swa_checkpoint(
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
            )

    wall_total = time.time() - wall_start
    print(f"\n  Training wall-clock: {wall_total/3600:.2f}h ({wall_total:.0f}s)")
    try:
        footprint_info.update(
            {
                "training_wall_clock_s": wall_total,
                "training_wall_clock_h": round(wall_total / 3600, 3),
                "num_epochs": args.num_epochs,
                "best_score": best_score,
            }
        )
        fp_path = os.path.join(args.output_dir, "footprint.json")
        with open(fp_path, "w") as fp:
            json.dump(footprint_info, fp, indent=2)
        print(f"  Footprint saved -> {fp_path}")
    except Exception as fe:
        print(f"  Footprint save failed: {fe}")

    print(f"\n  Done! best_score={best_score:.2f}")
    if not args.test_at_end:
        return

    run_final_test(args, best_ckpt, model, ema, xai_batch, device)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)
