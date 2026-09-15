from __future__ import annotations

import inspect
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from Model.Data.trajectories_dataset.dataset import TrajectoryDataset, seq_collate


def _find_tcnd_root(path: str) -> str:
    path = os.path.abspath(path)
    check = path
    for _ in range(6):
        if os.path.exists(os.path.join(check, "Best_track_data")):
            return check
        parent = os.path.dirname(check)
        if parent == check:
            break
        check = parent

    return path


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _build_dataset_kwargs(args, test: bool, test_year) -> dict:
    if isinstance(args.__dict__.get("_path_config", None), dict):
        dset_type = args._path_config.get("type", "test" if test else "train")
    else:
        dset_type = "test" if test else "train"

    ds_sig = inspect.signature(TrajectoryDataset.__init__)
    kwargs: dict = dict(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=getattr(args, "skip", 1),
        threshold=getattr(args, "threshold", 0.002),
        min_ped=getattr(args, "min_ped", 1),
        delim=getattr(args, "delim", " "),
        other_modal=getattr(args, "other_modal", "gph"),
        is_test=test,
    )
    if "split" in ds_sig.parameters:
        kwargs["split"] = dset_type
    elif "type" in ds_sig.parameters:
        kwargs["type"] = dset_type

    if "test_year" in ds_sig.parameters and test_year is not None:
        kwargs["test_year"] = test_year
    if "filter_region" in ds_sig.parameters:
        kwargs["filter_region"] = getattr(args, "filter_region", False)
    if "min_pct_in_scs" in ds_sig.parameters:
        kwargs["min_pct_in_scs"] = getattr(args, "min_pct_in_scs", 15.0)

    return kwargs, dset_type


def data_loader(
    args,
    path_config,
    test: bool = False,
    test_year: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    for_training: bool = False,
) -> tuple:

    if isinstance(path_config, dict):
        raw_path = path_config.get("root", "")
        dset_type_hint = path_config.get("type", "test" if test else "train")
    else:
        raw_path = str(path_config)
        dset_type_hint = "test" if test else "train"

    root = _find_tcnd_root(raw_path)
    print(f"DataLoader | root={root} | type={dset_type_hint} | year={test_year}")

    args._path_config = path_config
    ds_kwargs, dset_type = _build_dataset_kwargs(args, test, test_year)
    dataset = TrajectoryDataset(data_dir=root, **ds_kwargs)

    num_workers = getattr(args, "num_workers", 0)
    effective_batch_size = batch_size or args.batch_size

    loader_kwargs = dict(
        batch_size=effective_batch_size,
        shuffle=not test,
        collate_fn=seq_collate,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=_cuda_available(),
    )

    if for_training:
        is_train = dset_type == "train" and not test
        loader_kwargs["drop_last"] = is_train and len(dataset) > effective_batch_size
        loader_kwargs["persistent_workers"] = num_workers > 0
        loader_kwargs["prefetch_factor"] = 2 if num_workers > 0 else None
        loader_kwargs["pin_memory"] = _cuda_available() and num_workers > 0
        loader_kwargs["worker_init_fn"] = _worker_init_fn if num_workers > 0 else None

        _seed = seed if seed is not None else getattr(args, "seed", None)
        if _seed is not None and not test:
            gen = torch.Generator()
            gen.manual_seed(int(_seed))
            loader_kwargs["generator"] = gen

    loader = DataLoader(dataset, **loader_kwargs)

    print(
        f"  {len(dataset)} sequences loaded"
        + (
            f"  (workers={num_workers}, drop_last={loader_kwargs['drop_last']})"
            if for_training
            else ""
        )
    )
    return dataset, loader
