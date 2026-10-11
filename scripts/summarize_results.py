from __future__ import annotations

import json

from common.data import repo_path
from common.logging_utils import save_json


def load(path):
    p = repo_path(path)
    return json.loads(p.read_text()) if p.exists() else None  # missing results are left empty


def main():
    t1 = {n: load(f"results/task1_dpo/{n}_eval.json") for n in ["base", "standard"]}
    t2 = {n: load(f"results/task2_ppo/{n}_eval.json") for n in ["midpoint", "standard"]}
    t3 = {n: load(f"results/task3_grpo/{n}_eval.json") for n in ["midpoint", "standard"]}
    safety = load("results/task4_safety/safety_summary.json")
    feedback = load("results/task5_feedback/feedback_comparison.json")

    def pick(e, keys):
        return {k: e.get(k) for k in keys} if e else None

    rl_keys = ["reward_mean", "reward_se", "kl_per_token", "kl_per_sequence", "entropy_per_token", "length_mean", "length_std", "eos_rate"]
    policies = {  # the four task 4 policies with their own task metrics
        "sft": {"task1_eval": pick(t1["base"], ["reward_mean", "length_mean", "truncation_rate", "reference_preference_accuracy"])},
        "dpo": {"task1_eval": pick(t1["standard"], ["dpo_loss", "preference_accuracy", "kl_per_sequence", "reward_mean", "length_mean", "truncation_rate"])},
        "ppo": {"task2_eval": pick(t2["standard"], rl_keys), "task2_midpoint": pick(t2["midpoint"], rl_keys)},
        "grpo": {"task3_eval": pick(t3["standard"], rl_keys), "task3_midpoint": pick(t3["midpoint"], rl_keys)},
    }
    if safety:
        for p, s in safety["policies"].items():
            policies[p]["safety"] = {k: s[k] for k in ["safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate",
                                                       "unsafe_justified_refusal_rate", "ambiguous_rate", "mean_response_tokens"]}
    out = {"policies": policies, "task5": feedback, "safety_audit": safety["manual_audit"] if safety else None}
    save_json("results/cross_task_summary.json", out)

    for p, d in policies.items():
        line = f"{p:5s}"
        for k, e in d.items():
            if e and "reward_mean" in e:
                line += f" | {k} reward {e['reward_mean']:.3f} len {e['length_mean']:.0f}"
        if d.get("safety"):
            s = d["safety"]
            line += f" | safe_ans {s['safe_answer_rate']:.2f} over_ref {s['safe_over_refusal_rate']:.2f} unsafe_comp {s['unsafe_compliance_rate']:.2f}"
        print(line)


if __name__ == "__main__":
    main()
