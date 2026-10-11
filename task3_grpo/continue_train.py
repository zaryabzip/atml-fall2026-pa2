from __future__ import annotations

import argparse
import json
import random

import torch
from torch.optim import AdamW

from common.checkpointing import load_checkpoint, save_checkpoint
from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, device_memory_gb, save_json, set_seed, wall_timer
from common.metrics import sample_entropy, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.evaluate import evaluate_adapter
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def _row_logprobs(policy, seq, attn, pw, resp):
    return torch.cat([response_token_logprobs(policy, seq[i : i + 1], attn[i : i + 1], pw, resp[i : i + 1])[0] for i in range(seq.shape[0])])  # one row at a time for memory


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", resume: bool = False):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, optimizer, tokenizer = bundle["policy"], bundle["optimizer"], bundle["tokenizer"]
    reward_model, reward_tokenizer = bundle["reward_model"], bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    k = int(cfg["num_generations"])
    max_len = int(cfg["max_completion_length"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])

    order = list(range(len(prompts)))
    random.Random(int(cfg["seed"])).shuffle(order)
    picked = [prompts[i] for i in order[: cfg["updates"]]]

    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"{run_name}_log.jsonl"
    save_json(results_dir / f"{run_name}_prompt_ids.json", [r["prompt_id"] for r in picked])

    params = trainable_parameters(policy)
    run_cfg = {"updates": int(cfg["updates"]), "loss_type": loss_type}
    ckpt_path = out / "checkpoint.pt"
    ckpt_every = int(cfg.get("checkpoint_every", 0))

    start_step, time_offset, peak_gb = 0, 0.0, 0.0
    if resume and ckpt_path.exists():
        state = load_checkpoint(ckpt_path, {"policy": policy}, {"policy": optimizer})
        if state["run_cfg"] != run_cfg:
            raise ValueError(f"checkpoint {ckpt_path} is from a different run {state['run_cfg']}")
        start_step, time_offset, peak_gb = state["step"], state["elapsed_s"], state["peak_gb"]
        if log_path.exists():
            lines = [l for l in log_path.read_text().splitlines() if l.strip() and json.loads(l)["step"] <= start_step]
            log_path.write_text("".join(l + "\n" for l in lines))
        print(f"[{run_name}] resuming after update {start_step}", flush=True)
    else:
        log_path.unlink(missing_ok=True)
        ckpt_path.unlink(missing_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timer = wall_timer()
    elapsed = lambda: time_offset + timer()

    for step, row in enumerate(picked, 1):
        if step <= start_step:
            continue
        group_prompts = [prompt_messages(row)] * k
        with torch.no_grad():
            roll = batch_generate(
                policy,
                tokenizer,
                group_prompts,
                max_prompt_length=int(cfg["max_prompt_length"]),
                max_new_tokens=max_len,
                **cfg["generation"],
            )
            seq, attn, resp = (roll[key].clone() for key in ("sequences", "attention_mask", "response_ids"))  # clone out of inference mode
            pw = roll["prompt_width"]
            mask = roll["response_mask"].clone()
            old_logp = _row_logprobs(policy, seq, attn, pw, resp)
            with reference_mode(policy):
                ref_logp = _row_logprobs(policy, seq, attn, pw, resp)
            rewards = score_reward_pairs(reward_model, reward_tokenizer, group_prompts, roll["responses"], max_length=int(cfg["reward_max_length"])).float().cpu()
            adv = group_relative_advantages(rewards, torch.zeros(k, dtype=torch.long)).to(seq.device)
            token_mask = mask_truncated_sequences(mask, roll["truncated"]) if cfg["mask_truncated_completions"] else mask

        epoch_stats = []
        for _ in range(int(cfg["policy_epochs"])):
            new_logp = _row_logprobs(policy, seq, attn, pw, resp)
            loss, diag = grpo_policy_loss(new_logp, old_logp, adv, token_mask, ref_logp, eps, beta, loss_type, max_len)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non finite loss at update {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            peak_gb = max(peak_gb, device_memory_gb() or 0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
            optimizer.step()
            epoch_stats.append({"loss": loss.item(), "grad_norm": float(grad_norm), **{key: float(v) for key, v in diag.items()}})

        lengths = mask.sum(-1).tolist()
        kept = token_mask.sum(-1).tolist()
        denom = [max_len if loss_type == "dr_grpo" else max(n, 1.0) for n in kept]
        reward_std = float(rewards.std(unbiased=False))
        record = {
            "step": step,
            "prompt_id": row["prompt_id"],
            "reward": float(rewards.mean()),
            "reward_std": reward_std,
            "uninformative": reward_std <= 1e-6,
            "kl": float(sampled_kl(old_logp, ref_logp, mask)),
            "entropy": float(sample_entropy(old_logp, mask)),
            "response_length": float(sum(lengths) / len(lengths)),
            "masked_completions": int(sum(n == 0 for n in kept)),
            **{key: sum(e[key] for e in epoch_stats) / len(epoch_stats) for key in epoch_stats[0]},
            "rewards": rewards.tolist(),
            "advantages": adv.cpu().tolist(),
            "lengths": lengths,
            "truncated": [bool(t) for t in roll["truncated"]],
            "seq_weight": [abs(a) * n / d for a, n, d in zip(adv.cpu().tolist(), kept, denom)],
            "elapsed_s": elapsed(),
            "peak_memory_gb": peak_gb,
        }
        append_jsonl(log_path, record)
        print(
            f"[{run_name}] update {step}/{cfg['updates']} reward={record['reward']:.3f} std={reward_std:.3f} "
            f"kl={record['kl']:.4f} len={record['response_length']:.0f} masked={record['masked_completions']} {record['elapsed_s'] / 60:.1f}min",
            flush=True,
        )

        del roll, seq, attn, resp, mask, token_mask, old_logp, ref_logp, new_logp, loss
        clear_gpu()  # free cached memory

        if ckpt_every and step % ckpt_every == 0 and step < len(picked):
            save_checkpoint(ckpt_path, {"policy": policy}, {"policy": optimizer}, step=step, elapsed_s=elapsed(), peak_gb=peak_gb, run_cfg=run_cfg)

    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    ckpt_path.unlink(missing_ok=True)
    log = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]  # whole run including resumed parts
    summary = {
        "run_name": run_name,
        **run_cfg,
        "mean_reward": sum(r["reward"] for r in log) / len(log),
        "mean_within_group_reward_std": sum(r["reward_std"] for r in log) / len(log),
        "uninformative_group_fraction": sum(r["uninformative"] for r in log) / len(log),
        "generated_tokens": int(sum(sum(r["lengths"]) for r in log)),  # total sampled tokens across all updates
        "masked_completions": int(sum(r["masked_completions"] for r in log)),
        "wall_clock_s": elapsed(),
        "peak_memory_gb": peak_gb,
        "device": str(next(policy.parameters()).device),
        "output": str(out),
        "config": cfg,  # every hyperparameter used
    }
    save_json(results_dir / f"{run_name}_summary.json", summary)

    del policy, reward_model, optimizer, bundle
    clear_gpu()
    return summary


def run_fork(config_path: str, loss_type: str):
    cfg = load_yaml(config_path)
    name = f"fork_{loss_type}"
    out = repo_path(cfg["output"]).parent / name
    results_dir = repo_path(cfg["results_dir"])
    if (out / "adapter_config.json").exists():
        print(f"{name} already trained, skipping training", flush=True)
    else:
        run_grpo(config_path, str(out), int(cfg["fork_updates"]), loss_type, name, resume=True)
    if (results_dir / f"{name}_eval.json").exists():
        print(f"{name} already evaluated, reusing it", flush=True)
        metrics = json.loads((results_dir / f"{name}_eval.json").read_text())
    else:
        metrics = evaluate_adapter(config_path, str(out), name)
    log = [json.loads(l) for l in (results_dir / f"{name}_log.jsonl").read_text().splitlines() if l.strip()]
    return name, metrics, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.resume)


if __name__ == "__main__":
    main()
