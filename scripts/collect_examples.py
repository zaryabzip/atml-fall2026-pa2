"""Collect qualitative examples for every task into report/qualitative_examples.md (data only)."""
from __future__ import annotations

import json
from pathlib import Path

from common.data import read_jsonl

R = Path("results")
out: list[str] = []


def clip(t, n=700):
    t = (t or "").strip()
    return t if len(t) <= n else t[:n] + " [...]"


def block(title, text):
    out.append(f"**{title}**\n\n> " + clip(text).replace("\n", "\n> ") + "\n")


def jl(p):
    return json.loads(Path(p).read_text())


# Task 1: DPO vs base, plus reward-up-but-suspicious and word-limit cases
out.append("# Task 1: DPO\n")
std = jl(R / "task1_dpo/standard_qualitative.json")
for x in std[:2]:
    out.append(f"## Standard vs base (prompt {x['prompt_id'][:8]})\n")
    block("Prompt", x["prompt"])
    block(f"Base response (reward {x['base_reward']:.2f})", x["base_response"])
    block(f"DPO response (reward {x['trained_reward']:.2f})", x["trained_response"])
rq = jl(R / "task1_dpo/rq_support.json")["rq3_candidates"]
for x in rq["reward_up_but_suspicious"][:2]:
    out.append(f"## Reward up but suspicious (gain {x['reward_gain']:.2f}, {', '.join(x['flags'])})\n")
    block("Prompt", x["prompt"])
    block("Base response", x["base_response"])
    block("DPO response", x["dpo_response"])
for x in rq["word_limit_violations"][:2]:
    out.append(f"## Word limit broken ({x['model']}: {x['words']} words, limit {x['limit']})\n")
    block("Response", x["response"])

# Task 2 and 3: highest and lowest reward samples
for task, d, runs in [("Task 2: PPO", "task2_ppo", ["standard", "midpoint"]), ("Task 3: GRPO", "task3_grpo", ["standard", "fork_dr_grpo"])]:
    out.append(f"# {task}\n")
    for run in runs:
        q = jl(R / d / f"{run}_qualitative.json")
        for kind in ["highest_reward", "lowest_reward"]:
            x = q[kind][0]
            out.append(f"## {run}, {kind.replace('_', ' ')} (reward {x['reward']:.2f}, {x['length_tokens']} tokens, truncated={x['truncated']}, seq KL {x['seq_kl']:.3f})\n")
            block("Prompt", x["prompt"])
            block("Response", x["response"])

# Task 4: judge/class contradictions and unsafe prompts answered
out.append("# Task 4: safety\n")
pols = ["sft", "dpo", "ppo", "grpo"]
judged = {p: read_jsonl(R / f"task4_safety/judged_{p}.jsonl") for p in pols}
for cls, lab, why in [("UNSAFE", "SAFE_ANSWER", "unsafe prompt, judge says SAFE_ANSWER (possible harmful compliance)"),
                      ("SAFE", "JUSTIFIED_REFUSAL", "safe prompt, judge says JUSTIFIED_REFUSAL (possible over-refusal)")]:
    for p in pols:
        rows = [r for r in judged[p] if r["benchmark_class"] == cls and r["ai_label"] == lab]
        rows.sort(key=lambda r: (-(r.get("ai_confidence") or 0), r["xstest_id"]))
        out.append(f"## {p}: {why}; {len(rows)} cases, showing top by judge confidence\n")
        for r in rows[:1]:
            out.append(f"xstest_id {r['xstest_id']}, type {r['type']}, confidence {r.get('ai_confidence')}\n")
            block("Prompt", r["prompt"])
            block("Response", r["response"])
            block("Judge rationale", r.get("ai_rationale", ""))

# Task 5: verifier-vs-judge disagreement and controlled diagnostics
out.append("# Task 5: RLVR vs RLAIF\n")
dp = read_jsonl(R / "task5_feedback/diagnostic_pairs.jsonl")
diag = {(r["problem_id"], r.get("variant_type")): r for r in read_jsonl("data/task5_controlled_reward_diagnostics.jsonl")}
shown = 0
for r in dp:
    if r["verifier"] != r["judge"] and shown < 3:
        shown += 1
        out.append(f"## Diagnostic {r['problem_id']} ({r['kind']}): verifier={r['verifier']}, judge={r['judge']}\n")
        out.append(f"Variant: {r['variant']}\n")
for p in ["sft", "rlvr", "rlaif"]:
    f = R / f"task5_feedback/gsm_generated_{p}.jsonl"
    if not f.exists():
        continue
    rows = read_jsonl(f)
    for lab, rr in [("correct", [x for x in rows if x["correct"]]), ("no final answer", [x for x in rows if not x["format_ok"]])]:
        if rr:
            x = rr[0]
            out.append(f"## {p} GSM8K, {lab} (gold {x['gold_final']}, parsed {x['predicted_final']})\n")
            block("Question", x["question"])
            block("Response", x["response"])

Path("report/qualitative_examples.md").write_text("\n".join(out))
print("wrote report/qualitative_examples.md", len(out), "blocks")
