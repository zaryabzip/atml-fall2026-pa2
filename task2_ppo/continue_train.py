from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import (
    append_jsonl,
    device_memory_gb,
    get_rng_state,
    save_json,
    set_rng_state,
    set_seed,
    wall_timer,
)
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.evaluate import evaluate_adapter
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def _upcast_value_head(value_model):
    for p in value_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()  # fp16 adam breaks
    head = value_model.score if hasattr(value_model, "score") else value_model.classifier
    head.register_forward_pre_hook(lambda m, args: tuple(a.float() if torch.is_tensor(a) else a for a in args))


def _trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def _load_trainable(model, state):
    unexpected = model.load_state_dict(state, strict=False).unexpected_keys
    if unexpected:
        raise ValueError(f"checkpoint has unexpected keys {unexpected[:4]}")


def _save_checkpoint(path: Path, policy, value_model, policy_optimizer, value_optimizer, **meta):
    state = {
        "policy": _trainable_state(policy),
        "value": _trainable_state(value_model),
        "policy_optimizer": policy_optimizer.state_dict(),
        "value_optimizer": value_optimizer.state_dict(),
        "rng": get_rng_state(),
        **meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic save


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard", resume: bool = False):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    value_optimizer = bundle["value_optimizer"]
    policy_optimizer = bundle["policy_optimizer"]
    prompts = bundle["prompt_rows"]
    
    order = list(range(len(prompts)))
    random.Random(int(cfg["seed"])).shuffle(order)
    picked = [prompts[i] for i in order[: cfg["updates"]]]

    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"{run_name}_log.jsonl"
    save_json(results_dir / f"{run_name}_prompt_ids.json", [r["prompt_id"] for r in picked])

    _upcast_value_head(value_model)
    policy_params = trainable_parameters(policy)
    value_params = trainable_parameters(value_model)
    max_grad_norm = float(cfg["max_grad_norm"])
    run_cfg = {"updates": int(cfg["updates"]), "clip_epsilon": float(cfg["clip_epsilon"]), "kl_beta": float(cfg["kl_beta"])}
    ckpt_path = out / "checkpoint.pt"
    ckpt_every = int(cfg.get("checkpoint_every", 0))

    start_step, time_offset, peak_gb = 0, 0.0, 0.0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["run_cfg"] != run_cfg:
            raise ValueError(f"checkpoint {ckpt_path} is from a different run {ckpt['run_cfg']}")
        _load_trainable(policy, ckpt["policy"])
        _load_trainable(value_model, ckpt["value"])
        policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        value_optimizer.load_state_dict(ckpt["value_optimizer"])
        set_rng_state(ckpt["rng"])
        start_step, time_offset, peak_gb = ckpt["step"], ckpt["elapsed_s"], ckpt["peak_gb"]
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
        prompt = [prompt_messages(row)]
        with torch.no_grad():
            roll = batch_generate(
                policy,
                tokenizer,
                prompt,
                max_prompt_length=int(cfg["max_prompt_length"]),
                max_new_tokens=int(cfg["max_response_length"]),
                **cfg["generation"],
            )
            seq, attn, resp = (roll[k].clone() for k in ("sequences", "attention_mask", "response_ids"))  # clone out of inference mode
            pw = roll["prompt_width"]
            mask = roll["response_mask"].clone()
            old_logp, _ = response_token_logprobs(policy, seq, attn, pw, resp)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seq, attn, pw, resp)
            old_values = token_values(value_model, seq, attn)[:, pw - 1 : -1].float()
            raw_reward = score_reward_pairs(reward_model, reward_tokenizer, prompt, roll["responses"], max_length=int(cfg["reward_max_length"]))
            no_eos = (~torch.tensor(roll["terminated_with_eos"], device=raw_reward.device)).float()
            reward = raw_reward - float(cfg["missing_eos_penalty"]) * no_eos
            rewards = shaped_rewards(reward, old_logp, ref_logp, mask, cfg["kl_beta"])
            adv, returns = compute_gae(rewards, old_values, mask, cfg["gamma"], cfg["gae_lambda"])
            adv = normalize_advantages(adv, mask)

        epoch_stats = []
        for _ in range(int(cfg["ppo_epochs"])):
            new_logp, _ = response_token_logprobs(policy, seq, attn, pw, resp)
            policy_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, adv, mask, float(cfg["clip_epsilon"]))
            new_values = token_values(value_model, seq, attn)[:, pw - 1 : -1].float()
            value_loss = value_mse_loss(new_values, returns, mask)
            loss = policy_loss + float(cfg["value_coef"]) * value_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non finite loss at update {step} (policy {policy_loss.item()}, value {value_loss.item()})")

            policy_optimizer.zero_grad(set_to_none=True)
            value_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            peak_gb = max(peak_gb, device_memory_gb() or 0.0)
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy_params, max_grad_norm)
            value_grad_norm = torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm)
            policy_optimizer.step()
            value_optimizer.step()

            epoch_stats.append({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "clip_fraction": clip_frac.item(),
                "ratio_mean": float(masked_mean(ratio, mask)),
                "policy_grad_norm": float(policy_grad_norm),
                "value_grad_norm": float(value_grad_norm),
            })

        with torch.no_grad():
            valid = mask.bool()
            ret_var = returns[valid].var(unbiased=False)
            explained_var = 1.0 - (returns[valid] - old_values[valid]).var(unbiased=False) / ret_var.clamp_min(1e-8)

        record = {
            "step": step,
            "prompt_id": row["prompt_id"],
            "reward": float(reward.mean()),
            "raw_reward": float(raw_reward.mean()),
            "kl": float(sampled_kl(old_logp, ref_logp, mask)),
            "entropy": float(sample_entropy(old_logp, mask)),
            "response_length": int(roll["response_lengths"][0]),
            "terminated_with_eos": bool(roll["terminated_with_eos"][0]),
            "truncated": bool(roll["truncated"][0]),
            "value_explained_variance": float(explained_var),
            **{k: sum(e[k] for e in epoch_stats) / len(epoch_stats) for k in epoch_stats[0]},
            "clip_fraction_last_epoch": epoch_stats[-1]["clip_fraction"],
            "elapsed_s": elapsed(),
            "peak_memory_gb": peak_gb,
        }
        append_jsonl(log_path, record)
        print(
            f"[{run_name}] update {step}/{cfg['updates']} reward={record['reward']:.3f} kl={record['kl']:.4f} "
            f"clip={record['clip_fraction']:.3f} len={record['response_length']} {record['elapsed_s'] / 60:.1f}min",
            flush=True,
        )

        del roll, seq, attn, resp, mask, old_logp, ref_logp, old_values, new_logp, new_values, loss
        clear_gpu()  # free cached memory

        if ckpt_every and step % ckpt_every == 0 and step < len(picked):
            _save_checkpoint(
                ckpt_path, policy, value_model, policy_optimizer, value_optimizer,
                step=step, elapsed_s=elapsed(), peak_gb=peak_gb, run_cfg=run_cfg,
            )

    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    value_model.save_pretrained(str(out / "value"))
    ckpt_path.unlink(missing_ok=True)

    summary = {
        "run_name": run_name,
        **run_cfg,
        "wall_clock_s": elapsed(),
        "peak_memory_gb": peak_gb,
        "device": str(next(policy.parameters()).device),
        "output": str(out),
    }
    save_json(results_dir / f"{run_name}_summary.json", summary)

    del policy, value_model, reward_model, policy_optimizer, value_optimizer, bundle
    clear_gpu()
    return summary



def fork_name(clip_epsilon: float, kl_beta: float) -> str:
    return f"fork_eps{float(clip_epsilon)}_kl{float(kl_beta)}"


def run_fork(config_path: str, clip_epsilon: float, kl_beta: float):
    cfg = load_yaml(config_path)
    name = fork_name(clip_epsilon, kl_beta)
    out = repo_path(cfg["output"]).parent / name
    results_dir = repo_path(cfg["results_dir"])
    if (out / "adapter_config.json").exists():
        print(f"{name} already trained, skipping training", flush=True)
    else:
        run_ppo(config_path, str(out), int(cfg["fork_updates"]), clip_epsilon, kl_beta, name, resume=True)
    if (results_dir / f"{name}_eval.json").exists():
        print(f"{name} already evaluated, reusing it", flush=True)
        metrics = json.loads((results_dir / f"{name}_eval.json").read_text())
    else:
        metrics = evaluate_adapter(config_path, str(out), name)
    log = [json.loads(l) for l in (results_dir / f"{name}_log.jsonl").read_text().splitlines() if l.strip()]
    return name, metrics, log


def _stats(values):
    xs = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(xs.mean()), "std": float(xs.std(unbiased=False)), "max": float(xs.max())}


def training_stability(log):
    return {
        "updates": len(log),
        "policy_grad_norm": _stats([r["policy_grad_norm"] for r in log]),
        "policy_loss": _stats([r["policy_loss"] for r in log]),
        "clip_fraction": _stats([r["clip_fraction"] for r in log]),
        "ratio_abs_dev": _stats([abs(r["ratio_mean"] - 1.0) for r in log]),
        "kl": _stats([r["kl"] for r in log]),
        "final_kl": log[-1]["kl"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name, args.resume)


if __name__ == "__main__":
    main()
