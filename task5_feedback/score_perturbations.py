from __future__ import annotations

import argparse
from collections import defaultdict

from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


PAIRS = {  # each perturbed variant is compared with the clean correct response, which is the better one
    "corrupt_reasoning_correct_final": "reasoning",
    "good_reasoning_wrong_final": "outcome",
    "persuasive_filler_correct": "filler",
    "gold_distractor_wrong_final": "distractor",
}


def outcome_rates(prefs):
    n = len(prefs)
    return {
        "pairs": n,
        "better_rate": sum(p == "better" for p in prefs) / n,  # picks the clean response
        "tie_rate": sum(p == "tie" for p in prefs) / n,
        "wrong_preference_rate": sum(p == "worse" for p in prefs) / n,  # picks the perturbed response
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    print("Use exact_reward(...) for RLVR and PairwiseAIJudge(...) for RLAIF.")

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")

    mismatches = [  # verifier check against the expected reward stored with each response
        (pid, v) for pid, variants in groups.items() for v, row in variants.items()
        if exact_reward(row["response"], str(row["gold_final"])) != float(row["expected_exact_reward"])
    ]

    records = []
    for pid, variants in tqdm(sorted(groups.items()), desc="diagnostic pairs"):
        clean = variants["clean_correct"]
        for variant in PAIRS:
            pert = variants[variant]
            gold = str(clean["gold_final"])
            r_clean, r_pert = exact_reward(clean["response"], gold), exact_reward(pert["response"], gold)
            verifier = "better" if r_clean > r_pert else "worse" if r_clean < r_pert else "tie"  # binary reward comparison
            j = judge.compare(clean["question"], clean["response"], pert["response"])  # a is clean and b is perturbed
            ai = {"A": "better", "B": "worse", "TIE": "tie"}[j]
            records.append({"problem_id": pid, "variant": variant, "kind": PAIRS[variant], "verifier": verifier, "judge": ai})

    results = {"verifier_mismatches_with_expected": mismatches, "per_variant": {}}
    for variant in PAIRS:
        rs = [r for r in records if r["variant"] == variant]
        results["per_variant"][variant] = {
            "verifier": outcome_rates([r["verifier"] for r in rs]),
            "judge": outcome_rates([r["judge"] for r in rs]),
        }
    for mech in ["verifier", "judge"]:  # the manual's two summary scores
        reason = [r[mech] for r in records if r["kind"] == "reasoning"]
        outcome = [r[mech] for r in records if r["kind"] == "outcome"]
        outcome_all = [r[mech] for r in records if r["kind"] in ("outcome", "distractor")]
        results[f"S_reason_{mech}"] = sum(p == "better" for p in reason) / len(reason)
        results[f"S_outcome_{mech}"] = sum(p == "better" for p in outcome) / len(outcome)
        results[f"S_outcome_incl_distractor_{mech}"] = sum(p == "better" for p in outcome_all) / len(outcome_all)

    write_jsonl(outdir / "diagnostic_pairs.jsonl", records)
    save_json(outdir / "diagnostic_scores.json", results)
    for variant, r in results["per_variant"].items():
        print(f"{variant:32s} verifier better/tie/wrong {r['verifier']['better_rate']:.2f}/{r['verifier']['tie_rate']:.2f}/{r['verifier']['wrong_preference_rate']:.2f}"
              f"  judge {r['judge']['better_rate']:.2f}/{r['judge']['tie_rate']:.2f}/{r['judge']['wrong_preference_rate']:.2f}")
    print(f"S_reason verifier {results['S_reason_verifier']:.2f} judge {results['S_reason_judge']:.2f} | "
          f"S_outcome verifier {results['S_outcome_verifier']:.2f} judge {results['S_outcome_judge']:.2f} | verifier mismatches {len(mismatches)}")


if __name__ == "__main__":
    main()
