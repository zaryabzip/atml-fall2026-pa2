from __future__ import annotations

import argparse
import json
import math

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

OBSERVABLES = ["reward", "kl", "entropy", "response_length", "policy_loss", "clip_fraction"]
EVAL_RUNS = ["midpoint", "standard", "fork_eps0.05_kl0.1", "fork_eps0.2_kl0.1", "fork_eps0.5_kl0.1", "fork_eps0.2_kl0.0", "fork_eps0.2_kl0.2"]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def first_divergence(rdir, other, ref="fork_eps0.2_kl0.1", tol=1e-9):
    a = read_jsonl(rdir / f"{other}_log.jsonl")
    b = read_jsonl(rdir / f"{ref}_log.jsonl")  # same prompts and seed so updates line up one to one
    out = {}
    for key in OBSERVABLES:
        diffs = [abs(x[key] - y[key]) for x, y in zip(a, b)]
        first = next((i + 1 for i, d in enumerate(diffs) if d > tol), None)  # first update where the runs differ
        out[key] = {"first_update_differs": first, "abs_diff_per_update": diffs}
    out["response_text_first_differs"] = next((i + 1 for i, (x, y) in enumerate(zip(a, b)) if x["response_length"] != y["response_length"]), None)
    return out


def reward_informativeness(rdir):
    out = {}
    for run in EVAL_RUNS:
        s = read_jsonl(rdir / f"{run}_samples.jsonl")
        eos = [x for x in s if x["terminated_with_eos"]]
        out[run] = {
            "corr_reward_length": corr([x["length_tokens"] for x in s], [x["reward"] for x in s]),
            "corr_reward_length_finished": corr([x["length_tokens"] for x in eos], [x["reward"] for x in eos]),
            "corr_reward_seq_kl": corr([x["seq_kl"] for x in s], [x["reward"] for x in s]),  # does drift buy reward
            "reward_eos": float(np.mean([x["reward"] for x in eos])) if eos else None,
            "reward_no_eos": float(np.mean([x["reward"] for x in s if not x["terminated_with_eos"]])) if len(eos) < len(s) else None,
        }

    mid = {x["prompt_id"]: x for x in read_jsonl(rdir / "midpoint_samples.jsonl")}
    std = {x["prompt_id"]: x for x in read_jsonl(rdir / "standard_samples.jsonl")}
    keys = [k for k in std if k in mid]
    d_reward = np.array([std[k]["reward"] - mid[k]["reward"] for k in keys])  # same prompt and same per prompt seed
    d_len = np.array([std[k]["length_tokens"] - mid[k]["length_tokens"] for k in keys])
    same_text = sum(std[k]["response"] == mid[k]["response"] for k in keys)
    gains = sorted(keys, key=lambda k: -(std[k]["reward"] - mid[k]["reward"]))
    out["standard_vs_midpoint_paired"] = {
        "prompts": len(keys),
        "identical_responses": int(same_text),
        "mean_reward_change": float(d_reward.mean()),
        "reward_change_se": float(d_reward.std(ddof=1) / math.sqrt(len(keys))),
        "corr_reward_change_length_change": corr(d_len, d_reward),
        "biggest_reward_gains": [
            {"prompt_id": k, "reward_gain": std[k]["reward"] - mid[k]["reward"], "length_change": std[k]["length_tokens"] - mid[k]["length_tokens"],
             "prompt": std[k]["prompt"], "midpoint_response": mid[k]["response"], "standard_response": std[k]["response"]}
            for k in gains[:5]  # for checking by hand if the higher reward is a better answer
        ],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rdir = repo_path(cfg["results_dir"])

    clip = json.loads((rdir / "clipping_study.json").read_text())
    out = {
        "rq1_clipping": {
            "cached_batch": clip["cached_batch"]["per_epsilon"],
            "live_forks_identical": all(
                f["eval"]["reward_mean"] == clip["forks"]["0.2"]["eval"]["reward_mean"] for f in clip["forks"].values()
            ),
            "live_stability": {eps: f["stability"] for eps, f in clip["forks"].items()},
        },
        "rq2_kl_first_change": {
            "beta_0.0_vs_0.1": first_divergence(rdir, "fork_eps0.2_kl0.0"),
            "beta_0.2_vs_0.1": first_divergence(rdir, "fork_eps0.2_kl0.2"),
        },
        "rq3_reward_informativeness": reward_informativeness(rdir),
    }
    save_json(rdir / "rq_support.json", out)

    for name, d in out["rq2_kl_first_change"].items():
        print(name, {k: d[k]["first_update_differs"] for k in OBSERVABLES}, "text differs from update", d["response_text_first_differs"])
    for run, r in out["rq3_reward_informativeness"].items():
        if run != "standard_vs_midpoint_paired":
            print(f"{run:20s} corr(reward,len) {r['corr_reward_length']:.2f} finished {r['corr_reward_length_finished']:.2f} corr(reward,kl) {r['corr_reward_seq_kl']:.2f} eos {r['reward_eos']:.2f} no_eos {r['reward_no_eos']:.2f}")
    p = out["rq3_reward_informativeness"]["standard_vs_midpoint_paired"]
    print(f"standard vs midpoint paired reward change {p['mean_reward_change']:+.3f}±{p['reward_change_se']:.3f} identical responses {p['identical_responses']}/{p['prompts']}")


if __name__ == "__main__":
    main()
