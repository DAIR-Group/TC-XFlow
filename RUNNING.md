# Running TC-XFlow

## Setup

```bash
pip install torch numpy scipy matplotlib
pip install cartopy 
```

## Data

Point commands at a dataset root with storm-wise `train`/`val`/`test`
splits (best-track sequences, ERA5 patches, environmental descriptors),
in the layout expected by `Model/Data/loader.py`.

## Train

```bash
cd Train_scripts
python train_flowmatching.py \
    --dataset_root /path/to/TCND_vn \
    --output_dir runs/tcxflow_seed42 \
    --seed 42
```

Saves `best_model.pth`, `last_model.pth`, and periodic checkpoints under
`--output_dir`. See `train_fm/cli_args.py` for the full list of options
(loss weights, ablation switches, LR schedule). For multi-seed training,
repeat with different `--seed` / `--output_dir` values.

## Visualize

```bash
cd Train_scripts
python visual_evaluate_mode.py \
    --TC_data_path /path/to/TCND_vn \
    --tc_name WIPHA \
    --tc_date 2019073106 \
    --seed_checkpoints ../checkpoints/best_model_fm_seed0.pth \
                        ../checkpoints/best_model_fm_seed1.pth \
                        ../checkpoints/best_model_fm_seed2.pth \
    --output_dir outputs
```

Outputs a map with observed history, verifying track, and one forecast
line per seed (labeled with mean DPE).
