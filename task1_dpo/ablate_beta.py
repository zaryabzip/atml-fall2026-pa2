from __future__ import annotations

import argparse
from pathlib import Path

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    n = int(cfg["short_ablation_examples"])

    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])

    results = {}
    out_root = Path(cfg["standard_output"]).parent / "beta_ablation"

    for b in cfg["betas"]:
        name = f"beta_{b}"
        out = str(out_root / name)
        clear_gpu()
        if (repo_path(out) / "adapter_config.json").exists():
            print(f"beta={b} already trained, skipping training")
        else:
            print(f"Training DPO with beta={b} on {n} examples...")
            run_training(args.config, name, output_path=out, beta=b, max_examples=n, resume=True)
        eval_path = repo_path(cfg["results_dir"]) / f"{name}_eval.json"
        if eval_path.exists():
            print(f"beta={b} already evaluated, reusing {eval_path.name}")
            results[name] = {"beta": b, **load_json(eval_path)}
        else:
            results[name] = {"beta": b, **evaluate_adapter(args.config, out, name, beta=b)}

    save_json(Path(cfg["results_dir"]) / "beta_ablation.json", results)


if __name__ == "__main__":
    main()
