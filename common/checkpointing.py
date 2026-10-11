from __future__ import annotations

import os
from pathlib import Path

import torch

from common.logging_utils import get_rng_state, set_rng_state


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    unexpected = model.load_state_dict(state, strict=False).unexpected_keys
    if unexpected:
        raise ValueError(f"checkpoint has unexpected keys {unexpected[:4]}")


def save_checkpoint(path: Path, models: dict, optimizers: dict, **meta):
    state = {
        "models": {k: trainable_state(m) for k, m in models.items()},
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "rng": get_rng_state(),
        **meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic save


def load_checkpoint(path: Path, models: dict, optimizers: dict) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for k, m in models.items():
        load_trainable(m, state["models"][k])
    for k, o in optimizers.items():
        o.load_state_dict(state["optimizers"][k])
    set_rng_state(state["rng"])
    return state
