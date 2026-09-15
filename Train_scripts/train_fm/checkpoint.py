import torch


def unwrap(model):

    return model._orig_mod if hasattr(model, "_orig_mod") else model


def move(batch, device):

    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    return out


def save_checkpoint(
    path, epoch, model, opt, sched, best_score, ema=None, scaler=None, extra=None, model_cfg=None
):
    m = unwrap(model)
    ema_state = None
    if ema is not None:
        try:
            ema_state = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        except Exception:
            pass

    payload = {
        "epoch": epoch,
        "model": m.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.epoch,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_score": best_score,
        "best_dpe": best_score,
        "ema": ema_state,
        "model_cfg": model_cfg,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def resume_from_checkpoint(resume_path, model, opt, sched, scaler, ema, device):

    ck = torch.load(resume_path, map_location=device)
    unwrap(model).load_state_dict(ck["model"], strict=False)
    try:
        opt.load_state_dict(ck["optimizer"])
    except Exception as e:
        print(f"  WARN: Opt: {e}")
    sched.epoch = ck.get("scheduler", 0)
    start_ep = ck.get("epoch", 0) + 1
    best_score = ck.get("best_score", ck.get("best_dpe", float("inf")))
    patience_cnt = ck.get("patience_cnt", 0)
    if scaler and ck.get("scaler"):
        try:
            scaler.load_state_dict(ck["scaler"])
        except Exception:
            pass
    if ema and ck.get("ema"):
        for k, v in ck["ema"].items():
            if k in ema.shadow:
                ema.shadow[k].copy_(v.to(device))
    print(f"  Resumed ep{start_ep}  best={best_score:.1f}  patience={patience_cnt}")
    return start_ep, best_score, patience_cnt
