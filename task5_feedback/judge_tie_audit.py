from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import torch
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import append_jsonl, save_json
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge


@torch.no_grad()
def raw_compare(judge, problem, a, b):
    key = judge._key(problem, a, b)
    swap = int(key[:8], 16) % 2 == 1  # same a b swap rule as the repo judge
    aa, bb = (b, a) if swap else (a, b)
    text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt", add_generation_prompt=True).to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=judge.tokenizer.eos_token_id, eos_token_id=judge.tokenizer.eos_token_id)
    decoded = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", decoded)
    parsed = m.group(1) if m else None  # none means the repo would silently call it a tie
    result = parsed or "TIE"
    if swap:
        result = {"A": "B", "B": "A", "TIE": "TIE"}[result]
    return {"key": key, "raw": decoded, "parsed": parsed is not None, "result": result, "cached": judge.cache.get(key)}


def all_pairs(outdir, cfg):
    pairs = []
    for ds in ["gsm", "transfer"]:  # each policy against sft on the same problem
        gens = {p: read_jsonl(outdir / f"{ds}_generated_{p}.jsonl") for p in ["sft", "rlvr", "rlaif"]}
        for p in ["rlvr", "rlaif"]:
            for mine, base in zip(gens[p], gens["sft"]):
                pairs.append({"set": f"{ds}_{p}_vs_sft", "id": mine["id"], "problem": mine["question"], "a": mine["response"], "b": base["response"]})
    rows = read_jsonl(cfg["paths"]["task5_diagnostics"])
    by = {}
    for r in rows:
        by.setdefault(str(r["problem_id"]), {})[r["variant_type"]] = r
    for pid, v in sorted(by.items()):  # clean response against each perturbed one
        for variant in ["corrupt_reasoning_correct_final", "good_reasoning_wrong_final", "persuasive_filler_correct", "gold_distractor_wrong_final"]:
            pairs.append({"set": f"diagnostic_{variant}", "id": pid, "problem": v["clean_correct"]["question"], "a": v["clean_correct"]["response"], "b": v[variant]["response"]})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--summary-only", action="store_true", help="rebuild the summary from saved raw outputs")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    raw_path = outdir / "judge_raw_outputs.jsonl"
    done = {(r["set"], r["id"]) for r in read_jsonl(raw_path)} if raw_path.exists() else set()  # resume support

    if not args.summary_only:
        judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
        for pair in tqdm(all_pairs(outdir, cfg), desc="judge raw outputs"):
            if (pair["set"], pair["id"]) in done:
                continue
            rec = raw_compare(judge, pair["problem"], pair["a"], pair["b"])
            append_jsonl(raw_path, {"set": pair["set"], "id": pair["id"], **rec})

    rows = read_jsonl(raw_path)
    summary = {}
    for s in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == s]
        summary[s] = {
            "pairs": len(rs),
            "a_wins": sum(r["result"] == "A" for r in rs),
            "b_wins": sum(r["result"] == "B" for r in rs),
            "true_ties": sum(r["result"] == "TIE" and r["parsed"] for r in rs),  # judge actually said tie
            "unparseable": sum(not r["parsed"] for r in rs),  # repo counts these as ties
            "multi_label": sum(len(set(re.findall(r"\b(A|B|TIE)\b", r["raw"]))) > 1 for r in rs),  # like a then b, repo keeps the first
            "picked_first_shown": sum(bool(re.search(r"\b(A|B|TIE)\b", r["raw"])) and re.search(r"\b(A|B|TIE)\b", r["raw"]).group(1) == "A" for r in rs),  # position a in the prompt won
            "matches_cached_result": sum(r["result"] == r["cached"] for r in rs if r["cached"] is not None),
            "raw_examples_unparseable": [r["raw"] for r in rs if not r["parsed"]][:5],
            "raw_output_counts": dict(Counter(r["raw"] for r in rs).most_common(6)),
        }
    save_json(outdir / "judge_tie_audit.json", summary)
    for s, d in summary.items():
        print(f"{s:45s} A {d['a_wins']:3d} B {d['b_wins']:3d} tie {d['true_ties']:3d} unparseable {d['unparseable']:3d} multi {d['multi_label']:3d} first_shown {d['picked_first_shown']:3d} same_as_cache {d['matches_cached_result']}/{d['pairs']}")


if __name__ == "__main__":
    main()
