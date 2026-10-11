from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    ids_path = outdir / "manual_audit_ids.csv"
    if ids_path.exists():
        print("Audit IDs already exist, not overwriting:", ids_path)  # protects any labels already typed in
    else:
        pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(ids_path, index=False)
        print("Wrote fixed audit IDs:", ids_path)
    print("Join these IDs to each policy's generated responses and label without viewing AI labels first.")

    sheet_path = outdir / "manual_audit_sheet.csv"
    if sheet_path.exists():
        print("Labeling sheet already exists, not overwriting:", sheet_path)
        return
    rows = []
    for name in ["sft", "dpo", "ppo", "grpo"]:  # one row per audited prompt per policy
        gen = outdir / f"generated_{name}.jsonl"
        if not gen.exists():
            print(f"skipping {name}, no generated responses yet")
            continue
        by_id = {r["xstest_id"]: r for r in read_jsonl(gen)}
        for i in ids:
            r = by_id[i]
            rows.append({
                "xstest_id": i,
                "policy": name,
                "benchmark_class": r["benchmark_class"],
                "type": r["type"],
                "prompt": r["prompt"],
                "response": r["response"],
                "manual_label": "",  # fill with one of the five labels
            })
    pd.DataFrame(rows).to_csv(sheet_path, index=False)  # no ai labels in this sheet
    print(f"Wrote labeling sheet with {len(rows)} rows:", sheet_path)
    print("Labels to use are SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")


if __name__ == "__main__":
    main()
