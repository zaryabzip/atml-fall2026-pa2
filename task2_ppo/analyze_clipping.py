from __future__ import annotations

import argparse
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import save_json
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer
from task2_ppo.continue_train import run_fork, training_stability
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _pad(seqs):
    out = torch.zeros(len(seqs), max(len(s) for s in seqs))
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.as_tensor(s, dtype=torch.float32)
    return out


@torch.no_grad()
def cached_batch_study(cfg, rows):
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    device = next(policy.parameters()).device
    prompts = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}

    new_logps = []
    for r in rows:
        rendered = tokenizer.apply_chat_template(prompt_messages(prompts[r["prompt_id"]]), tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]))["input_ids"]  # same as batch_generate
        resp_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
        if r["terminated_with_eos"]:
            resp_ids = resp_ids + [tokenizer.eos_token_id]
        if len(resp_ids) != len(r["old_logprobs"]):
            raise ValueError(f"cached response {r['prompt_id']} does not re-tokenize to {len(r['old_logprobs'])} tokens")
        seq = torch.tensor([prompt_ids + resp_ids], device=device)
        logp, _ = response_token_logprobs(policy, seq, torch.ones_like(seq), len(prompt_ids), torch.tensor([resp_ids], device=device))
        new_logps.append(logp[0].float().cpu())
        clear_gpu()  # free cached memory
    del policy
    clear_gpu()

    mask = _pad([[1.0] * len(r["old_logprobs"]) for r in rows])
    old = _pad([r["old_logprobs"] for r in rows])
    ref = _pad([r["ref_logprobs"] for r in rows])
    values = _pad([r["values"] for r in rows])
    new = _pad(new_logps)
    task_reward = torch.tensor([r["effective_terminal_reward"] for r in rows], dtype=torch.float32)

    rewards = shaped_rewards(task_reward, old, ref, mask, cfg["kl_beta"])
    adv, returns = compute_gae(rewards, values, mask, cfg["gamma"], cfg["gae_lambda"])
    adv = normalize_advantages(adv, mask)
    ratio = torch.exp(new - old)

    results = {
        "rollouts": len(rows),
        "valid_tokens": int(mask.sum()),
        "ratio_mean": float(masked_mean(ratio, mask)),
        "ratio_abs_dev_mean": float(masked_mean((ratio - 1.0).abs(), mask)),
        "ratio_max": float(ratio[mask.bool()].max()),
        "ratio_min": float(ratio[mask.bool()].min()),
        "unclipped_surrogate": float(masked_mean(ratio * adv, mask)),
        "per_epsilon": {},
    }
    for eps in cfg["clip_values"]:
        eps = float(eps)
        loss, _, clip_frac = ppo_policy_loss(new, old, adv, mask, eps)
        affected = (((ratio > 1.0 + eps) & (adv > 0)) | ((ratio < 1.0 - eps) & (adv < 0))).float()
        results["per_epsilon"][str(eps)] = {
            "clipped_surrogate": float(-loss),
            "clip_fraction": float(clip_frac),
            "affected_token_fraction": float(masked_mean(affected, mask)),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])

    results_dir = repo_path(cfg["results_dir"])
    cached = cached_batch_study(cfg, rows)
    save_json(results_dir / "clipping_cached_batch.json", cached)
    for eps, r in cached["per_epsilon"].items():
        print(f"eps={eps} clip_fraction={r['clip_fraction']:.4f} affected={r['affected_token_fraction']:.4f} surrogate={r['clipped_surrogate']:.4f}", flush=True)

    forks = {}
    for eps in cfg["clip_values"]:
        name, metrics, log = run_fork(args.config, float(eps), float(cfg["kl_beta"]))
        forks[str(float(eps))] = {"run": name, "eval": metrics, "stability": training_stability(log)}
    save_json(results_dir / "clipping_study.json", {"cached_batch": cached, "forks": forks})


if __name__ == "__main__":
    main()
