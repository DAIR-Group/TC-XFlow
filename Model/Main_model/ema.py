from __future__ import annotations

import torch

from Model.Main_model.loss import unwrap_compiled


_HORIZON_EMA_BUFFER_NAMES = {"reg_dist_ema", "heading_err_ema"}


class EMAModel:
    def __init__(self, model, decay: float = 0.995):
        self.decay = decay
        raw_model = unwrap_compiled(model)
        self.shadow = {
            name: param.detach().clone()
            for name, param in raw_model.state_dict().items()
            if param.dtype.is_floating_point
            and name.rsplit(".", 1)[-1] not in _HORIZON_EMA_BUFFER_NAMES
        }

    def update(self, model):
        raw_model = unwrap_compiled(model)
        with torch.no_grad():
            for name, param in raw_model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
      
        raw_model = unwrap_compiled(model)
        backup, state_dict = {}, raw_model.state_dict()
        for name in self.shadow:
            if name not in state_dict:
                continue
            backup[name] = state_dict[name].detach().clone()
            state_dict[name].copy_(self.shadow[name])
        return backup

    def restore(self, model, backup):
        raw_model = unwrap_compiled(model)
        state_dict = raw_model.state_dict()
        for name, value in backup.items():
            if name in state_dict:
                state_dict[name].copy_(value)
