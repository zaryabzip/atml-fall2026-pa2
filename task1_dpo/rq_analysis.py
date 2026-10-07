from __future__ import annotations

import argparse
import json
import math

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer

RUNS = ["base", "standard", "beta_0.03", "beta_0.1", "beta_0.3"]
FORKS = ["beta_0.03", "beta_0.1", "beta_0.3"]


def trend(values):
    d = np.diff(values)  # step to step change across increasing beta
    if np.all(d > 0):
        return "increasing"
    if np.all(d < 0):
        return "decreasing"
    return "not monotonic"


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def beta_section(rdir):
    table = {}
    for run in RUNS:
        e = json.loads((rdir / f"{run}_eval.json").read_text())
        s = read_jsonl(rdir / f"{run}_samples.jsonl")
        n = e["eval_examples"]
        fin = [x for x in s if not x["truncated"]]  # answers that ended before the cap
        table[run] = {
            "dpo_loss": e["dpo_loss"],
            "preference_accuracy": e["preference_accuracy"],
            "accuracy_se": math.sqrt(e["preference_accuracy"] * (1 - e["preference_accuracy"]) / n),  # binomial standard error
            "log_ratio_margin": e["log_ratio_margin_mean"],
            "kl_per_token": e["kl_per_token"],
            "kl_per_sequence": e["kl_per_sequence"],
            "reward_mean": e["reward_mean"],
            "reward_se": e["reward_std"] / math.sqrt(n),
            "reward_finished": float(np.mean([x["reward"] for x in fin])) if fin else None,
            "reward_truncated": float(np.mean([x["reward"] for x in s if x["truncated"]])) if len(fin) < len(s) else None,
            "length_mean": e["length_mean"],
            "length_std": e["length_std"],
            "length_finished_mean": float(np.mean([x["length_tokens"] for x in fin])) if fin else None,
            "truncation_rate": e["truncation_rate"],
            "corr_length_reward_finished": corr([x["length_tokens"] for x in fin], [x["reward"] for x in fin]),
        }

    def paired(a, b):
        sa = {x["prompt_id"]: x["reward"] for x in read_jsonl(rdir / f"{a}_samples.jsonl")}
        sb = {x["prompt_id"]: x["reward"] for x in read_jsonl(rdir / f"{b}_samples.jsonl")}
        d = np.array([sa[k] - sb[k] for k in sa if k in sb])  # same prompt in both runs
        return {"mean_diff": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))), "n": int(len(d))}

    paired_rewards = {
        "standard_minus_base": paired("standard", "base"),
        **{f"{f}_minus_base": paired(f, "base") for f in FORKS},
        "beta_0.3_minus_beta_0.03": paired("beta_0.3", "beta_0.03"),
    }
    monotonic = {k: trend([table[f][k] for f in FORKS]) for k in ["preference_accuracy", "log_ratio_margin", "kl_per_sequence", "reward_mean", "length_mean"]}

    curves = {}
    for run in ["standard", "length_balanced", *FORKS]:
        log = read_jsonl(rdir / f"{run}_train_log.jsonl")
        thirds = np.array_split(np.arange(len(log)), 3)  # early middle late parts of training
        curves[run] = {
            "steps": len(log),
            "loss_by_third": [float(np.mean([log[i]["loss"] for i in t])) for t in thirds],
            "train_acc_by_third": [float(np.mean([log[i]["preference_accuracy"] for i in t])) for t in thirds],
            "chosen_reward_by_third": [float(np.mean([log[i]["chosen_reward"] for i in t])) for t in thirds],
            "rejected_reward_by_third": [float(np.mean([log[i]["rejected_reward"] for i in t])) for t in thirds],
            "grad_clipped_fraction": float(np.mean([r["grad_norm"] > 1.0 for r in log])),
        }
    return {"table": table, "monotonic_over_beta": monotonic, "paired_reward_differences": paired_rewards, "training_curves": curves}


def dataset_length_section(cfg, rdir):
    tok = load_tokenizer(cfg["base_model"])
    strat = read_jsonl(cfg["paths"]["dpo_length_eval"])
    longer = [abs(r["length_difference"]) for r in strat if r["length_stratum"] != "length_matched"]
    matched = [abs(r["length_difference"]) for r in strat if r["length_stratum"] == "length_matched"]
    out = {"length_eval_stratum_rule": {"min_abs_diff_in_longer_strata": min(longer), "max_abs_diff_in_matched": max(matched)}}
    cut = max(matched)  # pairs differing by more than this count as longer

    def stats(key, filtered_name):
        rows = read_jsonl(cfg["paths"][key])
        drop_file = rdir / f"filtered_{filtered_name}.json"
        dropped = set(json.loads(drop_file.read_text())["dropped_prompt_ids"]) if drop_file.exists() else set()
        rows = [r for r in rows if r.get("prompt_id") not in dropped]  # same rows the runs used
        diffs = []
        for r in rows:
            c, j = preference_responses(r)
            diffs.append(len(tok(c, add_special_tokens=False)["input_ids"]) - len(tok(j, add_special_tokens=False)["input_ids"]))  # chosen minus rejected tokens
        d = np.array(diffs)
        return {
            "pairs": int(len(d)),
            "chosen_longer_fraction": float((d > 0).mean()),
            "preferred_longer_share": float((d > cut).mean()),
            "length_matched_share": float((np.abs(d) <= cut).mean()),
            "rejected_longer_share": float((d < -cut).mean()),
            "mean_diff_tokens": float(d.mean()),
            "median_diff_tokens": float(np.median(d)),
        }

    out["standard_train"] = stats("dpo_standard_train", "dpo_standard_train")
    out["standard_eval"] = stats("dpo_standard_eval", "standard_eval")
    out["length_balanced_train"] = stats("dpo_length_train", "dpo_length_balanced_train")
    return out


def disagreement_candidates(rdir, k=8):
    std = {x["prompt_id"]: x for x in read_jsonl(rdir / "standard_samples.jsonl")}
    base = {x["prompt_id"]: x for x in read_jsonl(rdir / "base_samples.jsonl")}
    pairs = []
    for pid, s in std.items():
        b = base.get(pid)
        if b is None:
            continue
        gain = s["reward"] - b["reward"]
        flags = []
        if s["truncated"] and not b["truncated"]:
            flags.append("dpo answer cut off")
        if s["length_tokens"] > b["length_tokens"] + 50:
            flags.append("dpo answer much longer")
        if gain > 0.5 and flags:  # higher reward but a warning sign worth reading
            pairs.append({"prompt_id": pid, "reward_gain": gain, "flags": flags, "prompt": s["prompt"],
                          "dpo_response": s["response"], "base_response": b["response"]})
    pairs.sort(key=lambda x: -x["reward_gain"])
    wl = read_jsonl(rdir / "word_limit_responses.jsonl")
    base_wl = json.loads((rdir / "word_limit_base.json").read_text())["responses"] if (rdir / "word_limit_base.json").exists() else []
    violations = [{"model": r.get("model", "base"), "prompt_id": r["prompt_id"], "limit": r["word_limit"], "words": r["words"], "response": r["response"]}
                  for r in wl + [{**x, "model": "base"} for x in base_wl] if r["compliant"] is False]  # answers over their word limit
    return {"reward_up_but_suspicious": pairs[:k], "word_limit_violations": violations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rdir = repo_path(cfg["results_dir"])
    out = {
        "rq1_beta": beta_section(rdir),
        "rq2_dataset_length": dataset_length_section(cfg, rdir),
        "rq3_candidates": disagreement_candidates(rdir),
    }
    save_json(rdir / "rq_support.json", out)
    for run, r in out["rq1_beta"]["table"].items():
        print(f"{run:10s} acc {r['preference_accuracy']:.3f}±{r['accuracy_se']:.3f} reward {r['reward_mean']:.3f}±{r['reward_se']:.3f} "
              f"finished {r['reward_finished']:.3f} cut {r['reward_truncated']:.3f} corr_len_reward {r['corr_length_reward_finished']:.2f}")
    print("trend over beta:", out["rq1_beta"]["monotonic_over_beta"])
    print("paired reward diffs:", {k: f"{v['mean_diff']:+.3f}±{v['se']:.3f}" for k, v in out["rq1_beta"]["paired_reward_differences"].items()})
    for k in ["standard_train", "standard_eval", "length_balanced_train"]:
        d = out["rq2_dataset_length"][k]
        print(f"{k:22s} chosen longer {d['chosen_longer_fraction']:.3f} strata pref/matched/rej {d['preferred_longer_share']:.2f}/{d['length_matched_share']:.2f}/{d['rejected_longer_share']:.2f} mean diff {d['mean_diff_tokens']:+.1f}")
    print("rq3 candidates", len(out["rq3_candidates"]["reward_up_but_suspicious"]), "word limit violations", len(out["rq3_candidates"]["word_limit_violations"]))


if __name__ == "__main__":
    main()
