from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import preference_accuracy, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import filter_long_prompts, make_collate

EVAL_BATCH_SIZE = 1
RM_BATCH_SIZE = 4
N_QUALITATIVE = 5


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    rows = filter_long_prompts(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tokenizer, cfg, label="standard eval")
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "policy": load_policy(cfg, adapter_path=None if adapter == "none" else adapter, trainable=False),  # none means base model
    }


def _batches(items, size, desc):
    return tqdm([items[s : s + size] for s in range(0, len(items), size)], desc=desc)


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
def preference_metrics(policy, tokenizer, rows, cfg, beta: float):
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(policy.parameters()).device
    pol_c, pol_r, ref_c, ref_r = [], [], [], []
    for batch in _batches(rows, EVAL_BATCH_SIZE, "preference scoring"):
        chosen, rejected = collate(batch)
        chosen = {k: v.to(device) for k, v in chosen.items()}
        rejected = {k: v.to(device) for k, v in rejected.items()}
        pol_c.append(response_sequence_logprobs(policy, chosen)[0].cpu())
        pol_r.append(response_sequence_logprobs(policy, rejected)[0].cpu())
        with reference_mode(policy):
            ref_c.append(response_sequence_logprobs(policy, chosen)[0].cpu())
            ref_r.append(response_sequence_logprobs(policy, rejected)[0].cpu())

    pol_c, pol_r, ref_c, ref_r = (torch.cat(x) for x in (pol_c, pol_r, ref_c, ref_r))
    margin = (pol_c - ref_c) - (pol_r - ref_r)
    return {
        "beta": beta,
        "dpo_loss": float(-F.logsigmoid(beta * margin).mean()),
        "preference_accuracy": float((margin > 0).float().mean()),
        "raw_policy_preference_accuracy": preference_accuracy(pol_c, pol_r),
        "reference_preference_accuracy": preference_accuracy(ref_c, ref_r),
        "log_ratio_margin_mean": float(margin.mean()),
    }


@torch.no_grad()
def generation_metrics(policy, tokenizer, rows, cfg):
    set_seed(int(cfg["seed"]))
    max_new = int(cfg["max_generation_tokens"])
    max_prompt = int(cfg["max_sequence_length"])  # no prompt is cut
    gen = cfg["generation"]

    kl_token_sum, token_count = 0.0, 0.0
    samples = []
    for batch in _batches(rows, EVAL_BATCH_SIZE, "generation"):
        prompts = [prompt_messages_from_preference(r) for r in batch]
        out = batch_generate(policy, tokenizer, prompts, max_prompt, max_new, **gen)
        args = (out["sequences"], out["attention_mask"], out["prompt_width"], out["response_ids"])
        pol_logp, _ = response_token_logprobs(policy, *args)
        with reference_mode(policy):
            ref_logp, _ = response_token_logprobs(policy, *args)
        mask = out["response_mask"]
        n_tokens = float(mask.sum())
        kl_token_sum += float(sampled_kl(pol_logp, ref_logp, mask)) * n_tokens
        token_count += n_tokens
        seq_kl = ((pol_logp - ref_logp) * mask).sum(-1)

        for i, row in enumerate(batch):
            samples.append({
                "prompt_id": row.get("prompt_id"),
                "prompt": row.get("prompt"),
                "messages": prompts[i],
                "response": out["responses"][i],
                "length_tokens": out["response_lengths"][i],
                "truncated": out["truncated"][i],
                "seq_kl": float(seq_kl[i]),
            })

    metrics = {
        "kl_per_token": kl_token_sum / max(token_count, 1.0),
        "kl_per_sequence": float(np.mean([s["seq_kl"] for s in samples])),
        **_length_stats([s["length_tokens"] for s in samples], "length"),
        "truncation_rate": float(np.mean([s["truncated"] for s in samples])),
    }
    return metrics, samples


@torch.no_grad()
def qualitative_examples(policy, tokenizer, rows, samples, cfg, n=N_QUALITATIVE):
    set_seed(int(cfg["seed"]))
    max_new = int(cfg["max_generation_tokens"])
    max_prompt = int(cfg["max_sequence_length"])  # no prompt is cut
    prompts = [prompt_messages_from_preference(r) for r in rows[:n]]
    with reference_mode(policy):
        base = batch_generate(policy, tokenizer, prompts, max_prompt, max_new, **cfg["generation"])
    return [
        {
            "prompt_id": s["prompt_id"],
            "prompt": s["prompt"],
            "messages": prompts[i],
            "base_response": base["responses"][i],
            "trained_response": s["response"],
        }
        for i, s in enumerate(samples[:n])
    ]


@torch.no_grad()
def score_rewards(rm, rm_tok, messages, responses):
    rewards = []
    for start in tqdm(range(0, len(responses), RM_BATCH_SIZE), desc="reward scoring"):
        end = start + RM_BATCH_SIZE
        rewards += score_reward_pairs(rm, rm_tok, messages[start:end], responses[start:end], max_length=1280).tolist()
    return rewards


def evaluate_adapter(config_path: str, adapter: str, name: str, max_examples: int | None = None, beta: float | None = None):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg, tokenizer, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    rows = bundle["rows"][:max_examples] if max_examples else bundle["rows"]
    beta = float(cfg["beta"] if beta is None else beta)

    metrics = {"name": name, "adapter": adapter, "eval_examples": len(rows)}
    metrics.update(preference_metrics(policy, tokenizer, rows, cfg, beta))
    gen_metrics, samples = generation_metrics(policy, tokenizer, rows, cfg)
    metrics.update(gen_metrics)
    qualitative = qualitative_examples(policy, tokenizer, rows, samples, cfg)
    del policy, bundle
    clear_gpu()

    rm, rm_tok = load_reward_model(cfg)
    rewards = score_rewards(rm, rm_tok, [s["messages"] for s in samples], [s["response"] for s in samples])
    for s, r in zip(samples, rewards):
        s["reward"] = r
    base_rewards = score_rewards(rm, rm_tok, [q["messages"] for q in qualitative], [q["base_response"] for q in qualitative])
    for q, s, r in zip(qualitative, samples, base_rewards):
        q["base_reward"], q["trained_reward"] = r, s["reward"]
    metrics["reward_mean"] = float(np.mean(rewards))
    metrics["reward_std"] = float(np.std(rewards))
    del rm
    clear_gpu()

    results_dir = repo_path(cfg["results_dir"])
    save_json(results_dir / f"{name}_eval.json", metrics)
    write_jsonl(results_dir / f"{name}_samples.jsonl", samples)
    save_json(results_dir / f"{name}_qualitative.json", qualitative)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--beta", type=float, help="beta the adapter was trained with (for held-out DPO loss); default: config beta")
    args = ap.parse_args()
    metrics = evaluate_adapter(args.config, args.adapter, args.name, args.max_examples, args.beta)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
