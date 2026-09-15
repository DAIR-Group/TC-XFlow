from __future__ import annotations

import torch

from Model.Main_model.loss import _unwrap


class EMAModel:
    _HORIZON_EMA_BUFFER_NAMES = {"reg_dist_ema", "heading_err_ema"}

    def __init__(self, model, decay: float = 0.995):
        self.decay = decay
        m = _unwrap(model)
        self.shadow = {
            k: v.detach().clone()
            for k, v in m.state_dict().items()
            if v.dtype.is_floating_point
            and k.rsplit(".", 1)[-1] not in self._HORIZON_EMA_BUFFER_NAMES
        }

    def update(self, model):
        m = _unwrap(model)
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):

        m = _unwrap(model)
        backup, sd = {}, m.state_dict()
        for k in self.shadow:
            if k not in sd:
                continue
            backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k])
        return backup

    def restore(self, model, backup):
        m = _unwrap(model)
        sd = m.state_dict()
        for k, v in backup.items():
            if k in sd:
                sd[k].copy_(v)
