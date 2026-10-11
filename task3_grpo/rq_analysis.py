from __future__ import annotations

import argparse
import json
import math

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def paired_vs_midpoint(rdir, run):
    mid = {x["prompt_id"]: x for x in read_jsonl(rdir / "midpoint_samples.jsonl")}
    new = {x["prompt_id"]: x for x in read_jsonl(rdir / f"{run}_samples.jsonl")}
    keys = [k for k in new if k in mid]
    d = np.array([new[k]["reward"] - mid[k]["reward"] for k in keys])  # same prompt and per prompt seed
    return {
        "prompts": len(keys),
        "identical_responses": int(sum(new[k]["response"] == mid[k]["response"] for k in keys)),
        "mean_reward_change": float(d.mean()),
        "reward_change_se": float(d.std(ddof=1) / math.sqrt(len(keys))),
        "mean_length_change": float(np.mean([new[k]["length_tokens"] - mid[k]["length_tokens"] for k in keys])),
    }


def instability_sources(log):
    gen = sum(sum(r["lengths"]) for r in log)
    k = len(log[0]["lengths"])
    used = sum(sum(n for n, t in zip(r["lengths"], r["truncated"]) if not t) for r in log)  # tokens that reach the loss
    grads = np.array([r["grad_norm"] for r in log])
    return {
        "updates": len(log),
        "completions": len(log) * k,
        "masked_completion_fraction": sum(r["masked_completions"] for r in log) / (len(log) * k),  # cut at the cap so dropped
        "generated_tokens": int(gen),
        "tokens_used_in_loss_fraction": used / max(gen, 1),
        "uninformative_group_fraction": float(np.mean([r["uninformative"] for r in log])),
        "reward_std_mean": float(np.mean([r["reward_std"] for r in log])),
        "reward_std_min": float(np.min([r["reward_std"] for r in log])),
        "grad_norm_mean": float(grads.mean()),
        "grad_norm_cv": float(grads.std() / grads.mean()) if grads.mean() > 0 else None,  # spread of update size
        "grad_clipped_fraction": float((grads > 1.0).mean()),
        "corr_grad_norm_reward_std": corr(grads, [r["reward_std"] for r in log]),
        "corr_grad_norm_masked": corr(grads, [r["masked_completions"] for r in log]),
        "clip_fraction_mean": float(np.mean([r["clip_fraction"] for r in log])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rdir = repo_path(cfg["results_dir"])

    gs = json.loads((rdir / "group_size_study.json").read_text())
    norm = json.loads((rdir / "normalization_study.json").read_text())
    std_log = read_jsonl(rdir / "standard_log.jsonl")

    out = {
        "rq1_group_size": {
            "cache_study": gs["per_k"],
            "live_standard_run_uninformative_fraction": float(np.mean([r["uninformative"] for r in std_log])),  # k equals 4 during training
        },
        "rq2_normalization": {
            lt: {
                "eval": {k: f["eval"][k] for k in ["reward_mean", "reward_se", "kl_per_sequence", "length_mean", "length_std", "eos_rate"]},
                "length_conditioned": f["length_conditioned"],
                "train_length_first_half": float(np.mean(f["trajectory"]["response_length"][: len(f["trajectory"]["response_length"]) // 2])),
                "train_length_second_half": float(np.mean(f["trajectory"]["response_length"][len(f["trajectory"]["response_length"]) // 2 :])),
                "paired_vs_midpoint": paired_vs_midpoint(rdir, f["run"]),
            }
            for lt, f in norm["forks"].items()
        },
        "rq3_instability": {
            "standard": instability_sources(std_log),
            **{lt: instability_sources(read_jsonl(rdir / f"{f['run']}_log.jsonl")) for lt, f in norm["forks"].items()},
        },
        "standard_vs_midpoint": paired_vs_midpoint(rdir, "standard"),
    }
    save_json(rdir / "rq_support.json", out)

    print("live uninformative fraction", out["rq1_group_size"]["live_standard_run_uninformative_fraction"])
    for lt, r in out["rq2_normalization"].items():
        print(f"{lt:8s} reward {r['eval']['reward_mean']:.3f}±{r['eval']['reward_se']:.3f} len {r['eval']['length_mean']:.0f} "
              f"long share {r['length_conditioned']['weight_share_long']:.3f} identical vs midpoint {r['paired_vs_midpoint']['identical_responses']}")
    s = out["rq3_instability"]["standard"]
    print(f"standard masked {s['masked_completion_fraction']:.2f} tokens used {s['tokens_used_in_loss_fraction']:.2f} uninformative {s['uninformative_group_fraction']:.2f} grad cv {s['grad_norm_cv']:.2f}")


if __name__ == "__main__":
    main()
