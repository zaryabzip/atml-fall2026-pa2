from __future__ import annotations

import argparse
import json

from common.data import load_yaml, repo_path
from common.logging_utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    def load(name):
        path = outdir / name
        if not path.exists():
            raise FileNotFoundError(f"missing {path}, run the earlier task 5 steps first")
        return json.loads(path.read_text())

    gsm, transfer, diag = load("gsm_eval.json"), load("transfer_eval.json"), load("diagnostic_scores.json")

    policies = {}
    for name in gsm["policies"]:  # in domain next to out of domain for each policy
        g, t = gsm["policies"][name], transfer["policies"][name]
        policies[name] = {
            "gsm_accuracy": g["exact_accuracy"],
            "transfer_accuracy": t["exact_accuracy"],
            "accuracy_drop": g["exact_accuracy"] - t["exact_accuracy"],
            "gsm_format": g["format_compliance"],
            "transfer_format": t["format_compliance"],
            "gsm_tokens": g["mean_response_tokens"],
            "transfer_tokens": t["mean_response_tokens"],
            "gsm_win_vs_sft": g.get("win_rate_vs_sft"),
            "transfer_win_vs_sft": t.get("win_rate_vs_sft"),
            "win_drop": (g["win_rate_vs_sft"] - t["win_rate_vs_sft"]) if "win_rate_vs_sft" in g else None,
            "gsm_verifier_judge_agreement": g.get("verifier_judge_agreement"),
            "transfer_verifier_judge_agreement": t.get("verifier_judge_agreement"),
        }

    diagnostic = {  # how each feedback source reacts to the controlled edits
        "S_reason": {"verifier": diag["S_reason_verifier"], "judge": diag["S_reason_judge"]},
        "S_outcome": {"verifier": diag["S_outcome_verifier"], "judge": diag["S_outcome_judge"]},
        "S_outcome_incl_distractor": {"verifier": diag["S_outcome_incl_distractor_verifier"], "judge": diag["S_outcome_incl_distractor_judge"]},
        "per_variant": diag["per_variant"],
    }
    save_json(outdir / "feedback_comparison.json", {"policies": policies, "diagnostic": diagnostic})

    for name, p in policies.items():
        win = "" if p["gsm_win_vs_sft"] is None else f" win gsm {p['gsm_win_vs_sft']:.3f} transfer {p['transfer_win_vs_sft']:.3f}"
        print(f"{name:6s} acc gsm {p['gsm_accuracy']:.3f} transfer {p['transfer_accuracy']:.3f} drop {p['accuracy_drop']:+.3f}{win}")
    print(f"S_reason verifier {diagnostic['S_reason']['verifier']:.2f} judge {diagnostic['S_reason']['judge']:.2f} | "
          f"S_outcome verifier {diagnostic['S_outcome']['verifier']:.2f} judge {diagnostic['S_outcome']['judge']:.2f}")


if __name__ == "__main__":
    main()
