from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import append_jsonl
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4, partial_path=None):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = read_jsonl(partial_path) if partial_path and Path(partial_path).exists() else []  # rows saved by an earlier run
    done = {r["xstest_id"] for r in records}
    for start in tqdm(range(0, len(df), batch_size), desc=f"{policy_name} generation"):
        chunk = df.iloc[start:start + batch_size]
        if all(int(x) in done for x in chunk["xstest_id"]):  # same batches as before so resumed output matches
            continue
        n_before = len(records)
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok, trunc in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"], gen["truncated"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
                "truncated": bool(trunc),
            })
        if partial_path:
            for rec in records[n_before:]:
                append_jsonl(partial_path, rec)  # save each batch as it finishes
    del model
    clear_gpu()  # free the policy before the next one loads
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", default=["sft", "dpo", "ppo", "grpo"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Policies:", args.policies)
    print("XSTest rows:", len(load_xstest(cfg)))

    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    order = [int(x) for x in load_xstest(cfg)["xstest_id"]]  # fixed csv order
    for name in args.policies:
        out_path = outdir / f"generated_{name}.jsonl"
        if out_path.exists():
            print(f"{name} already generated, skipping")
            continue
        adapter = policy_specs(cfg)[name]
        if adapter and not (repo_path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(f"{name} adapter missing at {adapter}, train it first")
        partial = outdir / f"generated_{name}.partial.jsonl"
        records = generate_for_policy(cfg, name, partial_path=partial)
        by_id = {r["xstest_id"]: r for r in records}
        write_jsonl(out_path, [by_id[i] for i in order])  # keep the csv prompt order
        partial.unlink(missing_ok=True)
        print(f"{name}: wrote {len(order)} responses to {out_path}")


if __name__ == "__main__":
    main()
