import math
from typing import List, Optional

import torch
import torch.optim as optim

from .checkpoint import unwrap


def build_optimizer(
    model,
    lr_velocity,
    lr_encoder,
    weight_decay,
    lr_logits_scale: float = 0.2,
    lr_extra_scale: float = 0.2,
):
    raw = unwrap(model)

    encoder_ids = {id(p) for p in raw.encoder.parameters()}
    velocity_ids = {id(p) for p in raw.velocity.parameters()}
    covered_ids = encoder_ids | velocity_ids

    softmax_logit_names = {"hard_score_weight_logits"}
    softmax_logit_params, rest_extra_params = [], []
    for name, p in raw.named_parameters():
        if id(p) in covered_ids:
            continue
        short = name.rsplit(".", 1)[-1]
        (softmax_logit_params if short in softmax_logit_names else rest_extra_params).append(p)

    groups = [
        {"params": list(raw.encoder.parameters()), "lr": lr_encoder, "name": "encoder"},
        {"params": list(raw.velocity.parameters()), "lr": lr_velocity, "name": "velocity"},
    ]
    if rest_extra_params:
        groups.append(
            {
                "params": rest_extra_params,
                "lr": lr_velocity * lr_extra_scale,
                "name": "learnable_extra",
            }
        )
        print(
            f"  [build_optimizer] learnable_extra group: {len(rest_extra_params)} tensors "
            f"({sum(p.numel() for p in rest_extra_params)} params) - "
            f"speed_correction/log_sigma*/score_* @ lr×{lr_extra_scale} "
        )
    if softmax_logit_params:
        groups.append(
            {
                "params": softmax_logit_params,
                "lr": lr_velocity * lr_logits_scale,
                "name": "softmax_logits",
            }
        )
        print(
            f"  [build_optimizer] softmax_logits group: {len(softmax_logit_params)} tensors "
            f"({sum(p.numel() for p in softmax_logit_params)} params) - "
            f"hard_score_weight_logits @ lr×{lr_logits_scale} "
        )

    return optim.AdamW(groups, weight_decay=weight_decay)


def get_lrs(opt):
    lr_enc = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "encoder")
    lr_vel = next(pg["lr"] for pg in opt.param_groups if pg.get("name") == "velocity")
    return lr_enc, lr_vel


class TwoGroupScheduler:

    def __init__(
        self,
        opt,
        warmup_epochs,
        total_epochs,
        lr_vel,
        lr_vel_min,
        freeze_end_ep,
        lr_enc_peak,
        encoder_warmup_epochs=5,
    ):
        self.opt = opt
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.lr_vel = lr_vel
        self.lr_vel_min = lr_vel_min
        self.freeze_end = freeze_end_ep
        self.lr_enc_peak = lr_enc_peak
        self.enc_warmup = encoder_warmup_epochs
        self.epoch = 0

        self._lr_ratio = {}
        for pg in self.opt.param_groups:
            name = pg.get("name")
            if name not in (None, "encoder"):
                self._lr_ratio[name] = pg["lr"] / lr_vel if lr_vel > 0 else 1.0

    def _cosine(self, ep_from, ep_to, lr_start, lr_end, ep):
        t = max(0.0, min(1.0, (ep - ep_from) / max(ep_to - ep_from, 1)))
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))

    def step(self):
        ep = self.epoch

        if ep < self.warmup:
            lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup - 1, 1))
        else:
            lr_vel = self._cosine(self.warmup, self.total, self.lr_vel, self.lr_vel_min, ep)

        if ep < self.freeze_end:
            lr_enc = 0.0
        elif ep < self.freeze_end + self.enc_warmup:
            lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
        else:
            lr_enc = self._cosine(
                self.freeze_end + self.enc_warmup, self.total, self.lr_enc_peak, self.lr_vel_min, ep
            )

        for pg in self.opt.param_groups:
            name = pg.get("name")
            if name == "encoder":
                pg["lr"] = lr_enc
            elif name in self._lr_ratio:
                pg["lr"] = lr_vel * self._lr_ratio[name]
            else:
                pg["lr"] = lr_vel

        self.epoch += 1
        return lr_vel, lr_enc


class SWAHandler:

    def __init__(self, swa_lr: float = 2e-6):
        self.swa_lr = swa_lr
        self.active = False
        self.start_ep = None
        self.n_updates = 0
        self.avg_state = {}

    def should_activate(
        self, dpe_history: List[float], window: int = 3, threshold: float = 1.5
    ) -> bool:
        if len(dpe_history) < window:
            return False
        return (dpe_history[-window] - dpe_history[-1]) < threshold

    def activate(self, model, opt, ep: int):
        self.active = True
        self.start_ep = ep
        for pg in opt.param_groups:
            pg["lr"] = self.swa_lr

        m = unwrap(model)
        excluded_suffixes = {
            "reg_dist_ema",
            "heading_err_ema",
            "reg_dist_ema_warmed",
            "heading_err_ema_warmed",
        }
        self.avg_state = {
            k: v.detach().clone().float()
            for k, v in m.state_dict().items()
            if v.dtype.is_floating_point and k.rsplit(".", 1)[-1] not in excluded_suffixes
        }
        self.n_updates = 1
        print(f"  *** SWA ACTIVATED @ ep{ep} (lr -> {self.swa_lr:.1e}) ***")

    def update(self, model):
        if not self.active:
            return
        sd = unwrap(model).state_dict()
        n = self.n_updates
        for k in self.avg_state:
            if k in sd:
                self.avg_state[k] = (n * self.avg_state[k] + sd[k].detach().float()) / (n + 1)
        self.n_updates += 1

    def apply_to_model(self, model):
        if not self.active or not self.avg_state:
            return
        sd = unwrap(model).state_dict()
        for k in self.avg_state:
            if k in sd:
                sd[k].copy_(self.avg_state[k].to(sd[k].device))

    def restore_from_backup(self, model, backup):
        sd = unwrap(model).state_dict()
        for k, v in backup.items():
            if k in sd:
                sd[k].copy_(v)

    def save_avg_state(
        self, path: str, epoch: int, best_score: float, extra: Optional[dict] = None, model_cfg=None
    ):
        payload = {
            "epoch": epoch,
            "model": self.avg_state,
            "best_score": best_score,
            "is_swa": True,
            "swa_updates": self.n_updates,
            "model_cfg": model_cfg,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
