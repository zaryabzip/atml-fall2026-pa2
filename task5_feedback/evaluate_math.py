from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


MAX_PROMPT_TOKENS = 256  # longest math prompt is 239 tokens so none are cut
BATCH_SIZE = 4


def row_id(row):
    return str(row.get("prompt_id") or row.get("source_index"))


def generate_policy(cfg, rows, tokenizer, name, out_path):
    partial = out_path.with_suffix(".partial.jsonl")
    records = read_jsonl(partial) if partial.exists() else []  # rows saved by an earlier run
    done = {r["id"] for r in records}
    model = load_frozen_policy(cfg, name)
    for start in tqdm(range(0, len(rows), BATCH_SIZE), desc=f"{name} generation"):
        chunk = rows[start : start + BATCH_SIZE]
        if all(row_id(r) in done for r in chunk):  # same batches as before so resumed output matches
            continue
        gen = batch_generate(
            model,
            tokenizer,
            [prompt_messages(r) for r in chunk],
            max_prompt_length=MAX_PROMPT_TOKENS,
            max_new_tokens=int(cfg["math_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,  # greedy so every policy is decoded the same way
        )
        for r, resp, n_tok, trunc in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
            rec = {
                "id": row_id(r),
                "policy": name,
                "question": r["question"],
                "gold_final": str(r["gold_final"]),
                "response": resp,
                "predicted_final": extract_designated_final(resp),  # number after the last #### or none
                "correct": exact_reward(resp, str(r["gold_final"])),
                "format_ok": extract_designated_final(resp) is not None,
                "response_tokens": int(n_tok),
                "truncated": bool(trunc),
            }
            records.append(rec)
            append_jsonl(partial, rec)  # save as we go
    del model
    clear_gpu()  # free the policy before the next one loads
    by_id = {r["id"]: r for r in records}
    ordered = [by_id[row_id(r)] for r in rows]  # keep the dataset order
    write_jsonl(out_path, ordered)
    partial.unlink(missing_ok=True)
    return ordered


def judge_vs_sft(judge, gens, name):
    outcomes = []
    for mine, base in zip(gens[name], gens["sft"]):
        pref = judge.compare(mine["question"], mine["response"], base["response"])  # a is the policy and b is sft
        score = {"A": 1.0, "B": 0.0, "TIE": 0.5}[pref]
        v = "A" if mine["correct"] > base["correct"] else "B" if mine["correct"] < base["correct"] else "TIE"  # verifier pick on the same pair
        outcomes.append({"id": mine["id"], "judge": pref, "score": score, "verifier": v})
    decisive = [o for o in outcomes if o["verifier"] != "TIE"]  # pairs where exactly one answer is right
    verifier_ties = [o for o in outcomes if o["verifier"] == "TIE"]  # both right or both wrong
    return {
        "verifier_tie_pairs": len(verifier_ties),
        "judge_decisive_when_verifier_ties": sum(o["judge"] != "TIE" for o in verifier_ties),  # judge separates what the verifier cannot
        "win_rate_vs_sft": float(np.mean([o["score"] for o in outcomes])),
        "wins": sum(o["judge"] == "A" for o in outcomes),
        "losses": sum(o["judge"] == "B" for o in outcomes),
        "ties": sum(o["judge"] == "TIE" for o in outcomes),
        "verifier_judge_agreement": float(np.mean([o["judge"] == o["verifier"] for o in outcomes])),
        "decisive_pairs": len(decisive),
        "judge_agrees_on_decisive": float(np.mean([o["judge"] == o["verifier"] for o in decisive])) if decisive else None,
        "pairs": outcomes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    print("Exact verifier available as task5_feedback.rlvr.exact_reward")
    print("Pairwise judge available as task5_feedback.rlaif.PairwiseAIJudge")

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    gens = {}
    for name in policy_specs(cfg):  # generate with one policy loaded at a time
        path = outdir / f"{args.dataset}_generated_{name}.jsonl"
        gens[name] = read_jsonl(path) if path.exists() else generate_policy(cfg, rows, tokenizer, name, path)

    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")  # judge loads only after all policies are freed
    results = {"dataset": args.dataset, "examples": len(rows), "policies": {}}
    for name, g in gens.items():
        lengths = np.array([r["response_tokens"] for r in g])
        entry = {
            "exact_accuracy": float(np.mean([r["correct"] for r in g])),
            "format_compliance": float(np.mean([r["format_ok"] for r in g])),
            "mean_response_tokens": float(lengths.mean()),
            "std_response_tokens": float(lengths.std()),
            "truncation_rate": float(np.mean([r["truncated"] for r in g])),
            "failure_types": {  # why the wrong answers were wrong
                "no_final_answer": sum(not r["correct"] and not r["format_ok"] and not r["truncated"] for r in g),
                "cut_off_at_limit": sum(not r["correct"] and not r["format_ok"] and r["truncated"] for r in g),
                "wrong_number": sum(not r["correct"] and r["format_ok"] for r in g),
            },
        }
        if name != "sft":
            pairwise = judge_vs_sft(judge, gens, name)
            write_jsonl(outdir / f"{args.dataset}_pairs_{name}_vs_sft.jsonl", pairwise.pop("pairs"))
            entry.update(pairwise)
        results["policies"][name] = entry
        print(f"{name:6s} acc {entry['exact_accuracy']:.3f} format {entry['format_compliance']:.3f} len {entry['mean_response_tokens']:.0f}"
              + (f" win_vs_sft {entry['win_rate_vs_sft']:.3f} agree {entry['verifier_judge_agreement']:.3f}" if name != "sft" else ""))
    save_json(outdir / f"{args.dataset}_eval.json", results)


if __name__ == "__main__":
    main()
