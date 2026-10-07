from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.metrics import parse_word_limit, word_count
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.evaluate import _length_stats, preference_metrics
from task1_dpo.train import filter_long_prompts, run_training

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def word_limit_analysis(policy, tokenizer, prompts, cfg):
    set_seed(int(cfg["seed"]))
    max_new = int(cfg["max_generation_tokens"])
    max_prompt = int(cfg["max_sequence_length"])  # no prompt is cut
    out = batch_generate(policy, tokenizer, [prompt_messages(r) for r in prompts], max_prompt, max_new, **cfg["generation"])

    rows = []
    for r, response, n_tokens in zip(prompts, out["responses"], out["response_lengths"]):
        limit = parse_word_limit(prompt_messages(r)[-1]["content"])
        words = word_count(response)
        rows.append({
            "prompt_id": r["prompt_id"],
            "word_limit": limit,
            "words": words,
            "length_tokens": n_tokens,
            "compliant": None if limit is None else words <= limit,
            "response": response,
        })
    scored = [x for x in rows if x["compliant"] is not None]
    summary = {
        "compliance_rate": float(np.mean([x["compliant"] for x in scored])),
        "mean_words": float(np.mean([x["words"] for x in scored])),
        "mean_overshoot_ratio": float(np.mean([x["words"] / x["word_limit"] for x in scored])),
        **_length_stats([x["words"] for x in rows], "words"),
        **_length_stats([x["length_tokens"] for x in rows], "length_tokens"),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--base-word-limit", action="store_true", help="only run the word limit prompts on the base model")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    word_prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])

    if args.base_word_limit:
        tokenizer = load_tokenizer(cfg["base_model"])
        policy = load_policy(cfg, trainable=False)
        summary, rows = word_limit_analysis(policy, tokenizer, word_prompts, cfg)
        save_json(repo_path(cfg["results_dir"]) / "word_limit_base.json", {"summary": summary, "responses": rows})
        print(f"[base] word_limit_compliance={summary['compliance_rate']:.2f} mean_words={summary['mean_words']:.1f}")
        return

    # train the length-balanced condition (skipped if the adapter already exists).
    if not (repo_path(cfg["length_output"]) / "adapter_config.json").exists():
        run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"], output_path=cfg["length_output"], resume=True)
    if not (repo_path(cfg["standard_output"]) / "adapter_config.json").exists():
        raise FileNotFoundError(f"Train the standard model first: {cfg['standard_output']}")

    tokenizer = load_tokenizer(cfg["base_model"])
    stratified = filter_long_prompts(stratified, tokenizer, cfg, label="length eval")
    results, responses = {}, []
    for name, adapter in [("standard", cfg["standard_output"]), ("length_balanced", cfg["length_output"])]:
        print(f"Evaluating {name} ({adapter})...")
        policy = load_policy(cfg, adapter_path=adapter, trainable=False)

        per_stratum = {}
        for stratum in STRATA:
            print(f"  stratum: {stratum}")
            rows = [r for r in stratified if r["length_stratum"] == stratum]
            per_stratum[stratum] = {"n": len(rows), **preference_metrics(policy, tokenizer, rows, cfg, float(cfg["beta"]))}
        gap = per_stratum["preferred_longer"]["preference_accuracy"] - per_stratum["rejected_longer"]["preference_accuracy"]

        print("  word-limit prompts")
        wl_summary, wl_rows = word_limit_analysis(policy, tokenizer, word_prompts, cfg)
        responses += [{"model": name, **x} for x in wl_rows]

        results[name] = {"per_stratum": per_stratum, "length_bias_gap": gap, "word_limit": wl_summary}
        print(f"[{name}] gap={gap:.3f} " + " ".join(f"{s}={per_stratum[s]['preference_accuracy']:.3f}" for s in STRATA)
              + f" word_limit_compliance={wl_summary['compliance_rate']:.2f}")

        del policy
        clear_gpu()

    results_dir = repo_path(cfg["results_dir"])
    save_json(results_dir / "length_analysis.json", results)
    write_jsonl(results_dir / "word_limit_responses.jsonl", responses)


if __name__ == "__main__":
    main()
