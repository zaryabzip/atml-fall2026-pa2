from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_long_prompts(rows, tokenizer, cfg, label: str = "rows"):
    budget = int(cfg["max_sequence_length"]) - 1
    kept = [
        r for r in rows
        if len(tokenizer.apply_chat_template(prompt_messages_from_preference(r), tokenize=True, add_generation_prompt=True)) <= budget
    ]
    print(f"{label}: kept {len(kept)}/{len(rows)} (dropped {len(rows) - len(kept)} with prompt > {budget} tokens)")
    kept_ids = {id(r) for r in kept}
    slug = label.replace(".jsonl", "").replace(" ", "_")
    save_json(repo_path(cfg["results_dir"]) / f"filtered_{slug}.json", {
        "rule": f"drop pairs whose prompt alone exceeds {budget} tokens; truncate responses from the right",
        "kept": len(kept),
        "dropped": len(rows) - len(kept),
        "dropped_prompt_ids": [r.get("prompt_id") for r in rows if id(r) not in kept_ids],
    })
    return kept


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    tokenizer = load_tokenizer(cfg["base_model"])
    rows = filter_long_prompts(read_jsonl(path), tokenizer, cfg, label=Path(path).name)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _get_rng_state():
    state = {"cpu": torch.get_rng_state(), "random": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state["cpu"])
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _save_checkpoint(path: Path, model, optimizer, **meta):
    state = {
        "model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "rng": _get_rng_state(),
        **meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic save


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None, resume: bool = False):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model, loader, optimizer = bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    accum = int(cfg["grad_accum_steps"])
    max_grad_norm = float(cfg["max_grad_norm"])
    params = trainable_parameters(model)
    device = next(model.parameters()).device

    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"{run_name}_train_log.jsonl"
    ckpt_path = output / "checkpoint.pt"
    ckpt_every = int(cfg.get("checkpoint_every", 0))

    start_epoch, start_batch, step, time_offset = 0, 0, 0, 0.0
    epoch_rng, resume_rng = None, None
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["beta"] != beta or ckpt["examples"] != len(bundle["rows"]):
            raise ValueError(f"Checkpoint {ckpt_path} is from a different run (beta/examples mismatch)")
        unexpected = model.load_state_dict(ckpt["model"], strict=False).unexpected_keys
        if unexpected:
            raise ValueError(f"Checkpoint has unexpected keys: {unexpected[:4]}")
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, start_batch, step = ckpt["epoch"], ckpt["batch"], ckpt["step"]
        time_offset, epoch_rng, resume_rng = ckpt["elapsed_s"], ckpt["epoch_rng"], ckpt["rng"]
        if log_path.exists():
            lines = [l for l in log_path.read_text().splitlines() if l.strip() and json.loads(l)["step"] <= step]
            log_path.write_text("".join(l + "\n" for l in lines))
        print(f"[{run_name}] resuming from step {step} (epoch {start_epoch}, micro-batch {start_batch})")
    else:
        log_path.unlink(missing_ok=True)
        ckpt_path.unlink(missing_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timer = wall_timer()
    elapsed = lambda: time_offset + timer()

    window = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(cfg["epochs"])):
        skip = start_batch if epoch == start_epoch else 0
        if resume_rng is not None and skip == 0:
            _set_rng_state(resume_rng)
            resume_rng = None
        elif resume_rng is not None:
            torch.set_rng_state(epoch_rng)  # replay this epoch's shuffle order
        epoch_rng = torch.get_rng_state()

        bar = tqdm(loader, desc=f"[{run_name}] epoch {epoch + 1}/{cfg['epochs']}")
        for i, (chosen, rejected) in enumerate(bar):
            if i < skip:
                continue
            if resume_rng is not None:
                _set_rng_state(resume_rng)
                resume_rng = None
            chosen = {k: v.to(device) for k, v in chosen.items()}
            rejected = {k: v.to(device) for k, v in rejected.items()}

            # reference policy = the same base model with the LoRA adapter disabled.
            with torch.no_grad(), reference_mode(model):
                ref_c = response_sequence_logprobs(model, chosen)[0]
                ref_r = response_sequence_logprobs(model, rejected)[0]
            pol_c = response_sequence_logprobs(model, chosen)[0]
            pol_r = response_sequence_logprobs(model, rejected)[0]

            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            (loss / accum).backward()
            window.append({
                "loss": loss.item(),
                **{k: v.item() for k, v in diag.items()},
                "chosen_reward": (beta * (pol_c - ref_c)).mean().item(),
                "rejected_reward": (beta * (pol_r - ref_r)).mean().item(),
            })

            if (i + 1) % accum == 0 or i + 1 == len(loader):
                # short final window: rescale so its gradient is a mean over the micro-batches it has.
                if len(window) < accum:
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(accum / len(window))
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                clear_gpu()  # free cached memory
                step += 1

                record = {"step": step, "epoch": epoch, "grad_norm": float(grad_norm), "elapsed_s": elapsed()}
                record.update({k: sum(w[k] for w in window) / len(window) for k in window[0]})
                append_jsonl(log_path, record)
                bar.set_postfix(step=step, loss=f"{record['loss']:.4f}", acc=f"{record['preference_accuracy']:.3f}")
                tqdm.write(
                    f"[{run_name}] step {step} loss={record['loss']:.4f} "
                    f"acc={record['preference_accuracy']:.3f} margin={record['logit_mean']:.4f}"
                )
                window = []

                if ckpt_every and step % ckpt_every == 0:
                    end_of_epoch = i + 1 == len(loader)
                    _save_checkpoint(
                        ckpt_path, model, optimizer,
                        epoch=epoch + 1 if end_of_epoch else epoch,
                        batch=0 if end_of_epoch else i + 1,
                        step=step, elapsed_s=elapsed(), epoch_rng=epoch_rng,
                        beta=beta, examples=len(bundle["rows"]),
                    )
                    tqdm.write(f"[{run_name}] checkpoint saved at step {step}")

    model.save_pretrained(str(output))
    bundle["tokenizer"].save_pretrained(str(output))
    ckpt_path.unlink(missing_ok=True)

    summary = {
        "run_name": run_name,
        "beta": beta,
        "examples": len(bundle["rows"]),
        "optimizer_steps": step,
        "wall_clock_s": elapsed(),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "output": str(output),
    }
    save_json(results_dir / f"{run_name}_train_summary.json", summary)

    del model, optimizer, bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in the output dir")
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples, args.resume)


if __name__ == "__main__":
    main()
