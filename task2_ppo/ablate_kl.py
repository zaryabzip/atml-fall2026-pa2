from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task2_ppo.continue_train import run_fork, training_stability

TRAJECTORY_KEYS = ["reward", "kl", "entropy", "response_length", "policy_loss", "value_loss", "clip_fraction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    forks = {}
    for kl in cfg["kl_values"]:
        name, metrics, log = run_fork(args.config, float(cfg["clip_epsilon"]), float(kl))
        forks[str(float(kl))] = {
            "run": name,
            "eval": metrics,
            "stability": training_stability(log),
            "trajectory": {k: [r[k] for r in log] for k in TRAJECTORY_KEYS},
        }
    save_json(repo_path(cfg["results_dir"]) / "kl_study.json", {"forks": forks})


if __name__ == "__main__":
    main()
