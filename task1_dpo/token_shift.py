from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import response_sequence_logprobs
from common.logging_utils import save_json
from common.models import load_policy, load_tokenizer, reference_mode
from task1_dpo.train import filter_long_prompts, make_collate

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


@torch.no_grad()
def pair_shifts(policy, tokenizer, rows, cfg):
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(policy.parameters()).device
    out = []
    for row in tqdm(rows, desc="token shifts"):
        chosen, rejected = collate([row])
        stats = {}
        for side, batch in (("pos", chosen), ("neg", rejected)):
            batch = {k: v.to(device) for k, v in batch.items()}
            _, pol_tok, mask = response_sequence_logprobs(policy, batch)
            with reference_mode(policy):
                _, ref_tok, _ = response_sequence_logprobs(policy, batch)
            delta = ((pol_tok - ref_tok) * mask)[0]  # per token log ratio, prompt masked out
            n = float(mask.sum())
            stats[side] = {"tokens": n, "sum": float(delta.sum()), "mean": float(delta.sum()) / n}
        out.append({"stratum": row["length_stratum"], "T_pos": stats["pos"]["tokens"], "T_neg": stats["neg"]["tokens"],
                    "shift_pos": stats["pos"]["mean"], "shift_neg": stats["neg"]["mean"],
                    "margin": stats["pos"]["sum"] - stats["neg"]["sum"]})
    return out


def summarize(pairs):
    g = lambda rows, k: np.array([r[k] for r in rows], dtype=float)
    out = {}
    for name, rows in [("all", pairs)] + [(s, [p for p in pairs if p["stratum"] == s]) for s in STRATA]:
        dT = g(rows, "T_pos") - g(rows, "T_neg")
        m = g(rows, "margin")
        entry = {
            "n": len(rows),
            "mean_shift_chosen": float(g(rows, "shift_pos").mean()),
            "mean_shift_rejected": float(g(rows, "shift_neg").mean()),
            "mean_margin": float(m.mean()),
            "mean_length_diff": float(dT.mean()),
        }
        if len(rows) > 2 and dT.std() > 0:
            entry["corr_margin_length_diff"] = float(np.corrcoef(dT, m)[0, 1])
            entry["slope_margin_on_length_diff"] = float(np.polyfit(dT, m, 1)[0])
        out[name] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--out", default="token_shift.json")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tokenizer = load_tokenizer(cfg["base_model"])
    rows = filter_long_prompts(read_jsonl(cfg["paths"]["dpo_length_eval"]), tokenizer, cfg, label="length eval")
    if args.max_pairs:
        rows = rows[:: max(1, len(rows) // args.max_pairs)][: args.max_pairs]  # spread over all strata
    results = {}
    for name, adapter in [("standard", cfg["standard_output"]), ("length_balanced", cfg["length_output"])]:
        policy = load_policy(cfg, adapter_path=adapter, trainable=False)
        pairs = pair_shifts(policy, tokenizer, rows, cfg)
        results[name] = summarize(pairs)
        print(name, results[name]["all"], flush=True)
        del policy
    save_json(repo_path(cfg["results_dir"]) / args.out, results)


if __name__ == "__main__":
    main()
