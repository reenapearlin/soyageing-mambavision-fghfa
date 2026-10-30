from contextlib import contextmanager
import torch


class ModelEMA:
    def __init__(self, model, decay=0.99):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self.num_updates = 0

        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        buffers = {name for name, _ in model.named_buffers()}
        self.tracked_names = sorted(trainable | buffers)

        state = model.state_dict()
        self.shadow = {
            name: state[name].detach().clone()
            for name in self.tracked_names
        }

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for name in self.tracked_names:
            source = state[name].detach()
            if source.shape != self.shadow[name].shape:
                raise RuntimeError(
                    f"EMA tensor shape changed for {name}: "
                    f"{tuple(self.shadow[name].shape)} -> {tuple(source.shape)}"
                )
            if torch.is_floating_point(source):
                self.shadow[name].lerp_(source, 1.0 - self.decay)
            else:
                self.shadow[name].copy_(source)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model):
        state = model.state_dict()
        for name in self.tracked_names:
            state[name].copy_(self.shadow[name])

    def model_state_dict(self):
        return {
            name: value.detach().clone()
            for name, value in self.shadow.items()
        }

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "tracked_names": self.tracked_names,
            "shadow": self.model_state_dict(),
        }

    @contextmanager
    def average_parameters(self, model):
        state = model.state_dict()
        original = {
            name: state[name].detach().clone()
            for name in self.tracked_names
        }
        self.copy_to(model)
        try:
            yield
        finally:
            state = model.state_dict()
            for name, value in original.items():
                state[name].copy_(value)
