from __future__ import annotations

import argparse
from collections import defaultdict
import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Students should decide and document exactly how prompts/completions are partitioned for the
    requested equal-generation comparison.
    """
    groups = []
    for pid, comps in by_prompt.items():
        for start in range(0, 8, k):  # consecutive generation_index blocks
            groups.append((pid, comps[start : start + k]))
    return groups


def group_stats(groups, tol=1e-6):
    rewards = torch.tensor([c["reward"] for _, g in groups for c in g], dtype=torch.float64)
    group_ids = torch.tensor([i for i, (_, g) in enumerate(groups) for _ in g])
    adv = group_relative_advantages(rewards, group_ids)
    stds = np.array([np.std([c["reward"] for c in g]) for _, g in groups])
    centered = np.concatenate([np.array([c["reward"] for c in g]) - np.mean([c["reward"] for c in g]) for _, g in groups])
    informative = stds > tol
    return {
        "groups": len(groups),
        "generations": int(rewards.numel()),
        "informative_rate": float(informative.mean()),
        "uninformative_fraction": float(1.0 - informative.mean()),
        "mean_within_group_std": float(stds.mean()),
        "var_centered_reward": float(centered.var()),
        "var_advantage": float(adv.var(unbiased=False)),
        "mean_abs_advantage": float(adv.abs().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])

    prompt_mean = {pid: float(np.mean([c["reward"] for c in comps])) for pid, comps in by_prompt.items()}
    cut = float(np.median(list(prompt_mean.values())))
    bins = {"hard": {p for p, m in prompt_mean.items() if m <= cut}, "easy": {p for p, m in prompt_mean.items() if m > cut}}

    results = {
        "partition_rule": "each prompt's 8 cached completions sorted by generation_index and split into consecutive blocks of K",
        "difficulty_rule": f"prompt mean reward over its 8 cached completions, median split at {cut:.4f} (hard is at or below)",
        "informative_tolerance": 1e-6,
        "clipped_completions": sum(c["clipped_at_max"] for comps in by_prompt.values() for c in comps),
        "per_k": {},
    }
    for k in cfg["group_sizes"]:
        groups = regroup_equal_generation_budget(by_prompt, int(k))
        entry = {"all": group_stats(groups)}
        for name, members in bins.items():
            entry[name] = group_stats([g for g in groups if g[0] in members])
        results["per_k"][str(k)] = entry
        a = entry["all"]
        print(f"K={k} groups={a['groups']} informative={a['informative_rate']:.3f} mean_std={a['mean_within_group_std']:.3f} "
              f"var_centered={a['var_centered_reward']:.3f} hard_inf={entry['hard']['informative_rate']:.3f} easy_inf={entry['easy']['informative_rate']:.3f}")
    save_json(repo_path(cfg["results_dir"]) / "group_size_study.json", results)


if __name__ == "__main__":
    main()
