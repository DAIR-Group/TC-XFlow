import argparse


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--dataset_root", default=" ")
    p.add_argument("--obs_len", default=8, type=int)
    p.add_argument("--pred_len", default=12, type=int)
    p.add_argument("--num_workers", default=2, type=int)
    p.add_argument("--other_modal", default="gph")
    p.add_argument("--delim", default=" ")
    p.add_argument("--skip", default=1, type=int)
    p.add_argument("--min_ped", default=1, type=int)
    p.add_argument("--threshold", default=0.002, type=float)
    p.add_argument(
        "--filter_region",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--min_pct_in_scs",
        default=15.0,
        type=float,
    )
    p.add_argument("--d_cond", default=256, type=int)
    p.add_argument("--d_model", default=256, type=int)
    p.add_argument("--nhead", default=8, type=int)
    p.add_argument("--num_dec_layers", default=4, type=int)
    p.add_argument("--dim_ff", default=512, type=int)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--unet_in_ch", default=13, type=int)
    p.add_argument("--sigma_min", default=0.06, type=float)
    p.add_argument("--sigma_max", default=0.15, type=float)
    p.add_argument("--sigma_decay_end", default=100, type=int)
    p.add_argument("--lambda_reg", default=0.2, type=float)
    p.add_argument("--lambda_heading", default=0.07, type=float)
    p.add_argument(
        "--use_curvature_score_train",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--lambda_momentum",
        default=0.0,
        type=float,
    )
    p.add_argument("--lambda_hard_reg", default=0.02, type=float)
    p.add_argument("--log_sigma_reg_min_clamp", default=-3.0, type=float)
    p.add_argument("--disable_horizon_nll", action="store_true", default=False)
    p.add_argument("--use_ot", default=True, action="store_true")
    p.add_argument("--no_ot", dest="use_ot", action="store_false")
    p.add_argument("--ot_epsilon", default=0.05, type=float)
    p.add_argument("--n_ensemble", default=20, type=int)
    p.add_argument("--sigma_inference", default=0.04, type=float)
    p.add_argument("--n_inference_steps", default=10, type=int)

    p.add_argument("--num_epochs", default=250, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--lr_logits_scale", default=0.2, type=float)
    p.add_argument(
        "--lr_extra_scale",
        default=0.2,
        type=float,
    )
    p.add_argument("--lr_min", default=1e-6, type=float)
    p.add_argument("--warmup_epochs", default=5, type=int)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--grad_clip", default=1.0, type=float)
    p.add_argument("--use_amp", action="store_true", default=False)
    p.add_argument("--use_ema", default=True, action="store_true")
    p.add_argument("--no_ema", dest="use_ema", action="store_false")

    p.add_argument("--freeze_encoder_epochs", default=10, type=int)
    p.add_argument("--encoder_warmup_epochs", default=5, type=int)
    p.add_argument("--lr_enc_peak", default=5e-5, type=float)

    p.add_argument("--val_freq", default=5, type=int)
    p.add_argument("--patience", default=40, type=int)
    p.add_argument("--min_ep", default=20, type=int)
    p.add_argument("--hard_val_threshold", default=0.35, type=float)
    p.add_argument("--hard_val_freq", default=10, type=int)

    p.add_argument("--swa_lr", default=2e-6, type=float)
    p.add_argument("--swa_window", default=3, type=int)
    p.add_argument("--swa_threshold", default=1.5, type=float)
    p.add_argument("--swa_min_ep", default=50, type=int)

    p.add_argument("--tta_test", default=True, action="store_true")
    p.add_argument("--n_tta", default=5, type=int)
    p.add_argument("--multiscale_test", default=True, action="store_true")

    p.add_argument("--output_dir", default="runs/fm_v26")
    p.add_argument("--gpu_num", default="0")
    p.add_argument("--resume", default=None)
    p.add_argument("--test_at_end", action="store_true", default=True)
    p.add_argument("--no_test", dest="test_at_end", action="store_false")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--disable_l_heading",
        action="store_true",
        default=False,
        help="Ablation: disable L_heading_ms",
    )
    p.add_argument(
        "--disable_l_calib",
        action="store_true",
        default=False,
        help="Ablation: disable L_calib ",
    )
    p.add_argument(
        "--disable_l_reg",
        action="store_true",
        default=False,
        help="Ablation: disable L_reg ",
    )
    p.add_argument(
        "--disable_aug_c",
        action="store_true",
        default=False,
        help="Ablation: disable AUG-C recurvature",
    )
    p.add_argument(
        "--disable_learned_weights",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--disable_hard_reg",
        action="store_true",
        default=False,
    )
    p.add_argument("--ablation_name", type=str, default="")
    return p.parse_args()
