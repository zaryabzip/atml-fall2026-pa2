from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.logging_utils import append_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import sample_entropy, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode

N_QUALITATIVE = 3


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def _length_stats(lengths, prefix):
    lengths = np.asarray(lengths, dtype=float)
    q25, q75 = np.percentile(lengths, [25, 75])
    return {
        f"{prefix}_mean": float(lengths.mean()),
        f"{prefix}_std": float(lengths.std()),
        f"{prefix}_median": float(np.median(lengths)),
        f"{prefix}_iqr": float(q75 - q25),
    }


@torch.no_grad()
def evaluate_adapter(config_path: str, adapter: str, name: str, max_examples: int | None = None):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg, tokenizer, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    rm, rm_tok = bundle["reward"]
    rows = bundle["rows"][:max_examples] if max_examples else bundle["rows"]

    results_dir = repo_path(cfg["results_dir"])
    partial_path = results_dir / f"{name}_samples.partial.jsonl"
    samples = read_jsonl(partial_path) if partial_path.exists() else []
    done = {s["prompt_id"] for s in samples}
    if done:
        print(f"{name}: resuming eval, {len(done)} of {len(rows)} prompts already done", flush=True)
    max_prompt = int(cfg["max_prompt_length"])
    max_new = int(cfg["eval_max_response_length"])
    penalty = float(cfg["missing_eos_penalty"])

    for i, row in enumerate(tqdm(rows, desc=f"{name} eval")):
        if row["prompt_id"] in done:
            continue
        set_seed(int(cfg["seed"]) + i)  # per prompt seed so a resumed eval matches
        prompt = [prompt_messages(row)]
        roll = batch_generate(policy, tokenizer, prompt, max_prompt, max_new, **cfg["generation"])
        args = (roll["sequences"], roll["attention_mask"], roll["prompt_width"], roll["response_ids"])
        pol_logp, _ = response_token_logprobs(policy, *args)
        with reference_mode(policy):
            ref_logp, _ = response_token_logprobs(policy, *args)
        mask = roll["response_mask"]
        n_tokens = float(mask.sum())
        reward = float(score_reward_pairs(rm, rm_tok, prompt, roll["responses"], max_length=int(cfg["reward_max_length"]))[0])
        eos = bool(roll["terminated_with_eos"][0])
        sample = {
            "prompt_id": row["prompt_id"],
            "prompt": prompt[0][-1]["content"],
            "response": roll["responses"][0],
            "length_tokens": int(roll["response_lengths"][0]),
            "terminated_with_eos": eos,
            "truncated": bool(roll["truncated"][0]),
            "reward": reward,
            "effective_reward": reward - (0.0 if eos else penalty),
            "seq_kl": float(((pol_logp - ref_logp) * mask).sum()),
            "n_tokens": n_tokens,
            "token_kl_mean": float(sampled_kl(pol_logp, ref_logp, mask)),
            "token_entropy_mean": float(sample_entropy(pol_logp, mask)),
        }
        samples.append(sample)
        append_jsonl(partial_path, sample)
        del roll, pol_logp, ref_logp, mask
        clear_gpu()  # free cached memory

    token_count = sum(s["n_tokens"] for s in samples)
    rewards = [s["reward"] for s in samples]
    metrics = {
        "name": name,
        "adapter": adapter,
        "eval_examples": len(samples),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_se": float(np.std(rewards) / math.sqrt(len(rewards))),
        "effective_reward_mean": float(np.mean([s["effective_reward"] for s in samples])),
        "kl_per_token": sum(s["token_kl_mean"] * s["n_tokens"] for s in samples) / max(token_count, 1.0),
        "kl_per_sequence": float(np.mean([s["seq_kl"] for s in samples])),
        "entropy_per_token": sum(s["token_entropy_mean"] * s["n_tokens"] for s in samples) / max(token_count, 1.0),
        **_length_stats([s["length_tokens"] for s in samples], "length"),
        "eos_rate": float(np.mean([s["terminated_with_eos"] for s in samples])),
        "truncation_rate": float(np.mean([s["truncated"] for s in samples])),
    }
    by_reward = sorted(samples, key=lambda s: s["reward"])
    qualitative = {"highest_reward": by_reward[-N_QUALITATIVE:][::-1], "lowest_reward": by_reward[:N_QUALITATIVE]}

    save_json(results_dir / f"{name}_eval.json", metrics)
    write_jsonl(results_dir / f"{name}_samples.jsonl", samples)
    save_json(results_dir / f"{name}_qualitative.json", qualitative)
    partial_path.unlink(missing_ok=True)

    del policy, rm, bundle
    clear_gpu()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    metrics = evaluate_adapter(args.config, args.adapter, args.name, args.max_examples)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
