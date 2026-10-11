from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task3_grpo.continue_train import run_fork

TRAJECTORY_KEYS = ["reward", "reward_std", "kl", "entropy", "response_length", "grad_norm", "clip_fraction"]


def length_conditioned(log, cut):
    lengths, weights = [], []
    for r in log:
        for n, w, t in zip(r["lengths"], r["seq_weight"], r["truncated"]):
            if not t:
                lengths.append(n)
                weights.append(w)
    lengths, weights = np.array(lengths), np.array(weights)
    share = weights / max(weights.sum(), 1e-12)
    short, long = lengths <= cut, lengths > cut
    return {
        "completions": int(len(lengths)),
        "weight_share_short": float(share[short].sum()),
        "weight_share_long": float(share[long].sum()),
        "mean_weight_short": float(weights[short].mean()) if short.any() else None,
        "mean_weight_long": float(weights[long].mean()) if long.any() else None,
        "corr_length_weight": float(np.corrcoef(lengths, weights)[0, 1]) if len(lengths) > 1 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    runs = {lt: run_fork(args.config, lt) for lt in ["grpo", "dr_grpo"]}
    all_lengths = [n for _, _, log in runs.values() for r in log for n, t in zip(r["lengths"], r["truncated"]) if not t]
    cut = float(np.median(all_lengths))

    out = {"length_split_rule": f"median kept completion length over both forks, {cut:.1f} tokens", "forks": {}}
    for lt, (name, metrics, log) in runs.items():
        out["forks"][lt] = {
            "run": name,
            "eval": metrics,
            "length_conditioned": length_conditioned(log, cut),
            "trajectory": {k: [r[k] for r in log] for k in TRAJECTORY_KEYS},
        }
        lc = out["forks"][lt]["length_conditioned"]
        print(f"{lt}: reward={metrics['reward_mean']:.3f} kl/tok={metrics['kl_per_token']:.5f} len={metrics['length_mean']:.0f} "
              f"long_share={lc['weight_share_long']:.3f} corr={lc['corr_length_weight']}")
    save_json(repo_path(cfg["results_dir"]) / "normalization_study.json", out)


if __name__ == "__main__":
    main()
