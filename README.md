# ATML PA2 - LLM Post-Training

Implementation and experiments for ATML PA2: DPO (Task 1), PPO (Task 2), GRPO (Task 3), safety calibration (Task 4), and RLVR vs RLAIF (Task 5). Every number in the report traces to a JSON/CSV file under `results/` produced by one of the commands below.

## 1. Setup

```bash
git clone https://github.com/zaryabzip/atml-fall2026-pa2.git
cd atml-fall2026-pa2
python -m pip install -r requirements.txt
python -m scripts.download_assets      # fixed course data, caches and checkpoints
python -m scripts.validate_assets
python -m scripts.check_environment
```

Course assets come from https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets (pinned revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`). Public base, reward and judge models download at runtime. Run every command from the repository root.

The code runs on CUDA, Apple MPS, or CPU (`common/models.py:get_device`). All reported runs used Apple Silicon (MPS, 16 GB unified memory), fp16, seed 6304.

## 2. Reproducing every experiment

Each task can be run stage by stage (commands below) or end to end with its pipeline script (`scripts/run_task<N>.sh`). The pipeline scripts run one job at a time, skip stages whose results already exist, and resume interrupted training from the last checkpoint. Long evaluations and generation steps also resume where they stopped.

### Task 1 - DPO (`configs/dpo.yaml`)

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard            # standard DPO, 1 epoch (add --resume to continue)
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name base   # untouched base policy, same protocol
python -m task1_dpo.ablate_beta --config configs/dpo.yaml                          # beta in {0.03, 0.10, 0.30}, 600-pair forks + evals
python -m task1_dpo.analyze_length --config configs/dpo.yaml                       # length-balanced model, strata, word limits
python -m task1_dpo.analyze_length --config configs/dpo.yaml --base-word-limit     # word-limit prompts on the base policy
python -m task1_dpo.rq_analysis --config configs/dpo.yaml                          # research-question tables from saved results
# or: scripts/run_task1.sh
```

### Task 2 - PPO (`configs/ppo.yaml`)

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard   # 20 updates from the supplied midpoint (--resume supported)
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml                     # cached batch + epsilon forks + evals
python -m task2_ppo.ablate_kl --config configs/ppo.yaml                            # beta_KL forks + evals
python -m task2_ppo.rq_analysis --config configs/ppo.yaml
# or: scripts/run_task2.sh
```

### Task 3 - GRPO (`configs/grpo.yaml`)

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard # 20 updates, K=4 (--resume supported)
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml                 # K in {2,4,8} from the supplied cache
python -m task3_grpo.compare_normalization --config configs/grpo.yaml              # grpo vs dr_grpo forks + evals
python -m task3_grpo.rq_analysis --config configs/grpo.yaml
# or: scripts/run_task3.sh
```

### Task 4 - Safety calibration (`configs/feedback.yaml`)

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # SFT, DPO, PPO, GRPO; greedy, 256 tokens
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml     # fixed 60 audit IDs + blind labeling sheet
python -m task4_safety.judge_responses --config configs/feedback.yaml      # fixed AI judge labels
# fill results/task4_safety/manual_audit_sheet.csv by hand, without looking at AI labels
python -m task4_safety.evaluate_safety --config configs/feedback.yaml      # rates, categories, manual-vs-AI agreement
# or: scripts/run_task4.sh
```

### Task 5 - RLVR vs RLAIF (`configs/feedback.yaml`)

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
python -m task5_feedback.judge_tie_audit --config configs/feedback.yaml    # separates real judge ties from unparseable outputs
# or: scripts/run_task5.sh
```

### Cross-task summary, figures and tables

```bash
python -m scripts.summarize_results    # results/cross_task_summary.json
python -m scripts.make_figures         # report/figures/*.pdf|png and report/tables/*.csv|md|tex
cd report && tectonic pa2_report.tex   # builds the report PDF (NeurIPS 2026 style)
```

## 3. Corrected objective defects

| Task | File | Released | Corrected |
|---|---|---|---|
| 1 | `task1_dpo/dpo.py` | `beta * (policy_margin + ref_margin)` | `beta * (policy_margin - ref_margin)`; accuracy uses the margin relative to the reference |
| 2 | `task2_ppo/ppo.py` | `torch.maximum(surr1, surr2)` | `torch.minimum(surr1, surr2)` (clipped surrogate) |
| 3 | `task3_grpo/grpo.py` | advantages normalized over the whole batch | mean and std computed within each prompt group |

## 4. Implementation choices

- **DPO sequence length:** follows the course announcement's prompt-preserving patch (`common/data.py`, upstream commit `076e7cf`). Prompts are never truncated and responses are truncated from the right. Pairs whose prompt alone does not fit in 768 tokens are dropped, using the same rule for every DPO train and eval run. Dropped prompt IDs are in `results/task1_dpo/filtered_*.json`.
- **PPO/GRPO prompts:** the training pool is shuffled with seed 6304, and update *k* uses the *k*-th prompt, so short forks use the same first prompts as the standard run. IDs used are in `results/task*/<run>_prompt_ids.json`. Generation follows the released `batch_generate` and configs.
- **PPO value head:** its trainable weights are kept in fp32 (fp16 AdamW produced NaN). The run stops on any non-finite loss.
- **Evaluation:** fixed eval sets and decoding from the configs. PPO/GRPO evals use the 768-token frozen-eval cap and per-prompt seeds (6304 + index). Task 4 and Task 5 use greedy decoding.
- **Checkpointing:** DPO every 5 optimizer steps, PPO/GRPO every update, with `--resume`.

## 5. Outputs

- `results/` - machine-readable logs, metrics and generated responses for every reported number (committed).
- `report/figures/`, `report/tables/` - figures and tables built from `results/` by `scripts/make_figures.py`.
- `outputs/` - trained adapters and checkpoints (not committed).

Compute for the standard continuations on Apple MPS: PPO 20 updates in 8.3 min (peak 12.6 GB); GRPO 20 updates in 14.9 min (peak 10.5 GB); standard DPO one epoch in 5.8 h. Peak memory on MPS is `torch.mps.driver_allocated_memory`.

## 6. Attribution

- **Course starter code** (data/model loaders, objective helpers, verifier, judges, asset scripts) and the course's prompt-preserving patch to `common/data.py` are by the ATML PA2 course staff.
- **Libraries:** PyTorch, Hugging Face Transformers, PEFT, NumPy, pandas, matplotlib.
- **Models:** Qwen2.5 (Alibaba Cloud) and `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`.
- **Data:** UltraFeedback, XSTest, GSM8K and SVAMP, through the fixed course subsets.
- No other external code was reused.
