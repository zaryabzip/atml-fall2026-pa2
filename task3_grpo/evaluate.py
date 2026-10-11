from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from common.rl_eval import evaluate_rl_policy


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_adapter(config_path: str, adapter: str, name: str, max_examples: int | None = None):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    rm, rm_tok = bundle["reward"]
    rows = bundle["rows"][:max_examples] if max_examples else bundle["rows"]
    metrics = evaluate_rl_policy(
        cfg,
        rows,
        bundle["tokenizer"],
        bundle["policy"],
        rm,
        rm_tok,
        name,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["eval_max_response_length"]),
        reward_max_length=int(cfg["reward_max_length"]),
    )
    metrics["adapter"] = adapter
    del rm, bundle
    clear_gpu()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    metrics = evaluate_adapter(args.config, args.adapter, args.name, args.max_examples)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
