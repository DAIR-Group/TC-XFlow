import time

import torch
from torch.amp import autocast

from Model.main_model import augment_batch

from train_fm.checkpoint import unwrap, move
from train_fm.schedulers import get_lrs


def train_one_epoch(model, trl, opt, scaler, swa, args, ep, freeze, nstep, device):
    model.train()
    sum_loss = sum_cfm = sum_reg = sum_head = sum_dpe1 = 0.0
    n_sanitized_batches = 0
    t0_ep = time.perf_counter()

    for i, batch in enumerate(trl):
        bl = move(list(batch), device)
        bl_aug = augment_batch(bl, disable_c=args.disable_aug_c)
        opt.zero_grad()
        with autocast(device_type="cuda", enabled=args.use_amp):
            bd = model.get_loss_breakdown(bl_aug, epoch=ep)

        total_has_grad = bd["total"].requires_grad and torch.isfinite(bd["total"])
        if not total_has_grad:
            n_sanitized_batches += 1
            print(
                f"  WARN: [{ep}][{i}] total loss had no valid gradient "
                f"(finite={torch.isfinite(bd['total']).item()} "
                f"requires_grad={bd['total'].requires_grad} "
                f"cfm={bd['l_cfm']:.4f} reg={bd['l_reg']:.4f} "
                f"h4s={bd['l_heading']:.4f} calib={bd['l_calib']:.4f} "
                f"score={bd['l_score']:.4f} hreg={bd.get('l_hard_reg', 0.0):.4f}) "
            )
            loss_for_backward = 0.0 * bd["_t_l_cfm"]
        else:
            loss_for_backward = bd["total"]

        scaler.scale(loss_for_backward).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if not torch.isfinite(grad_norm):
            n_sanitized_batches += 1
            bad_names = []
            for name, p in unwrap(model).named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    if len(bad_names) < 5:
                        bad_names.append(name)
            print(
                f"  WARN: [{ep}][{i}] non-finite grad_norm={grad_norm.item()} "
                f"sanitized_params(first 5)={bad_names[:5]} "
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if freeze:
            for p in unwrap(model).encoder.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        scaler.step(opt)
        scaler.update()
        model.ema_update()
        swa.update(model)

        sum_loss += bd["total"].item()
        sum_cfm += bd["l_cfm"]
        sum_reg += bd["l_reg"]
        sum_head += bd["l_heading"]
        sum_dpe1 += bd["ade_1step"]

        if i % 30 == 0:
            _, lr_vel = get_lrs(opt)
            enc_s = "frozen" if freeze else "active"
            swa_s = " [SWA]" if swa.active else ""
            print(
                f"  [{ep:>3}][{i:>3}/{nstep}]"
                f"  tot={bd['total'].item():.4f}"
                f"  cfm={bd['l_cfm']:.4f}"
                f"  reg={bd['l_reg']:.4f}"
                f"  h4s={bd['l_heading']:.4f}"
                f"  hreg={bd.get('l_hard_reg', 0.0):.4f}"
                f"  lam_d={bd['lam_dir']:.2f}"
                f"  dpe1={bd['ade_1step']:.0f}km"
                f"  enc={enc_s}{swa_s}"
                f"  lr={lr_vel:.2e}"
            )

    train_loss = sum_loss / nstep
    _, lr_vel_used = get_lrs(opt)
    sanitize_s = f"  sanitized={n_sanitized_batches}/{nstep}" if n_sanitized_batches > 0 else ""
    print(
        f"\n  -- Ep{ep:>3}"
        f"  train={train_loss:.6f}"
        f"  cfm={sum_cfm/nstep:.4f}"
        f"  reg={sum_reg/nstep:.4f}"
        f"  h4s={sum_head/nstep:.4f}"
        f"  dpe1={sum_dpe1/nstep:.0f}km"
        f"  lr={lr_vel_used:.2e}"
        f"  t={time.perf_counter()-t0_ep:.0f}s"
        f"{sanitize_s}"
    )
    return train_loss
