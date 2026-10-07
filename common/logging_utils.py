from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.data import repo_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> Path:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: str | Path, obj: Any) -> None:
    p = ensure_parent(path)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def wall_timer():
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


def get_rng_state() -> dict[str, Any]:
    state = {"cpu": torch.get_rng_state(), "random": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def device_memory_gb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1e9
    return None
