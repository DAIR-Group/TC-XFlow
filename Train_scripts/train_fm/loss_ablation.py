import math
import types

import torch


def apply_ablation_patch(raw, args):

    print(
        f"  [ABLATION] Disabled: "
        f"{'L_heading ' if args.disable_l_heading else ''}"
        f"{'L_calib ' if args.disable_l_calib else ''}"
        f"{'L_reg ' if args.disable_l_reg else ''}"
        f"{'AUG-C ' if args.disable_aug_c else ''}"
        f"{'Kendall_weights ' if args.disable_learned_weights else ''}"
        f"{'HardScoreReg ' if args.disable_hard_reg else ''}"
    )
    orig_get_loss_breakdown = raw.get_loss_breakdown.__func__
    ablation_flags = {
        "disable_l_heading": args.disable_l_heading,
        "disable_l_calib": args.disable_l_calib,
        "disable_l_reg": args.disable_l_reg,
        "disable_learned_weights": args.disable_learned_weights,
        "disable_hard_reg": args.disable_hard_reg,
    }

    def patched_get_loss_breakdown(self, batch_list, epoch=0, **kwargs):
        bd = orig_get_loss_breakdown(self, batch_list, epoch=epoch, **kwargs)

        l_cfm = bd["_t_l_cfm"]
        l_reg = bd["_t_l_reg"]
        l_heading = bd["_t_l_heading"]
        l_calib = bd["_t_l_calib"]
        l_hard_reg = bd["_t_l_hard_reg"]

        if ablation_flags["disable_l_reg"]:
            l_reg = l_reg * 0.0
        if ablation_flags["disable_l_heading"]:
            l_heading = l_heading * 0.0
        if ablation_flags["disable_l_calib"]:
            l_calib = l_calib * 0.0
        if ablation_flags["disable_hard_reg"]:
            l_hard_reg = l_hard_reg * 0.0

        half_log_2pi = 0.5 * math.log(2.0 * math.pi)
        if ablation_flags["disable_learned_weights"]:
            total = (
                l_cfm
                + bd["lam_reg"] * 0.20 * l_reg
                + bd["lam_dir"] * 0.07 * l_heading
                + bd["lam_calib"] * 0.10 * l_calib
            )
        else:
            prec_reg = torch.exp(-2.0 * self.log_sigma_reg.clamp(min=-3.0))
            prec_heading = torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))
            prec_calib = torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))
            total = (
                l_cfm
                + bd["lam_reg"]
                * (0.5 * prec_reg * l_reg + self.log_sigma_reg.clamp(min=-3.0) + half_log_2pi)
                + bd["lam_dir"]
                * (
                    0.5 * prec_heading * l_heading
                    + self.log_sigma_heading.clamp(min=-3.0)
                    + half_log_2pi
                )
                + bd["lam_calib"]
                * (0.5 * prec_calib * l_calib + self.log_sigma_calib.clamp(min=-3.0) + half_log_2pi)
            )

        total = total + self.lambda_hard_reg * l_hard_reg
        if not torch.isfinite(total):
            total = total.new_zeros(())

        bd.update(
            {
                "total": total,
                "l_reg": float(l_reg.detach()),
                "l_heading": float(l_heading.detach()),
                "l_calib": float(l_calib.detach()),
                "l_hard_reg": float(l_hard_reg.detach()),
            }
        )
        return bd

    raw.get_loss_breakdown = types.MethodType(patched_get_loss_breakdown, raw)
