from __future__ import annotations

import argparse


def get_args():
    p = argparse.ArgumentParser(description="TC-XFlow multi-seed track visualizer")
    p.add_argument("--TC_data_path", required=True)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--tc_name", default="WIPHA")
    p.add_argument("--tc_date", default="2019073106")
    p.add_argument("--dset_type", default="test")
    p.add_argument(
        "--test_year",
        type=int,
        default=None,
    )
    p.add_argument(
        "--filter_region",
        action="store_true",
        default=False,
    )
    p.add_argument("--min_pct_in_scs", type=float, default=15.0)
    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--ode_steps", type=int, default=10)
    p.add_argument("--num_ensemble", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--delim", default=" ")
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--min_ped", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.002)
    p.add_argument("--other_modal", default="gph")
    p.add_argument(
        "--seed_checkpoints",
        nargs="+",
        required=True,
    )
    return p.parse_args()
