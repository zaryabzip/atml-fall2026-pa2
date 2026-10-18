#!/bin/zsh

cd "${0:A:h}/.."  # run from the repo root
PY=.venv/bin/python  # venv python
CFG=configs/dpo.yaml  # task config
export PYTHONUNBUFFERED=1  # print logs right away

log() { echo "$(date '+%F %T') $*"; }  # timestamped log line

run_stage() {  # run one stage and stop the pipeline if it fails
  local name=$1; shift
  log "START $name: $*"
  "$@"  # run the stage command
  local code=$?
  if (( code != 0 )); then  # stage failed
    log "FAILED $name (exit $code). Pipeline stopped. Rerun scripts/run_task1.sh to continue."
    exit $code
  fi
  log "DONE $name"
}

while pgrep -f "task1_dpo\.(train|evaluate|ablate_beta|analyze_length)" >/dev/null; do  # wait until no other job is running
  log "waiting for a running Task 1 job to finish"
  sleep 300
done

if [[ -f outputs/task1_dpo/standard/adapter_config.json ]]; then  # skip training if the adapter is saved
  log "SKIP standard training (adapter exists)"
else
  run_stage "train standard" $PY -m task1_dpo.train --config $CFG --run-name standard --resume  # resumes from the last checkpoint
fi

if [[ -f results/task1_dpo/standard_eval.json ]]; then  # skip if results are saved
  log "SKIP standard evaluation (results exist)"
else
  run_stage "evaluate standard" $PY -m task1_dpo.evaluate --config $CFG --adapter outputs/task1_dpo/standard --name standard
fi

if [[ -f results/task1_dpo/beta_ablation.json ]]; then  # skip if results are saved
  log "SKIP beta ablation (results exist)"
else
  run_stage "beta ablation" $PY -m task1_dpo.ablate_beta --config $CFG
fi

if [[ -f results/task1_dpo/length_analysis.json ]]; then  # skip if results are saved
  log "SKIP length analysis (results exist)"
else
  run_stage "length analysis" $PY -m task1_dpo.analyze_length --config $CFG
fi

log "ALL TASK 1 RUNS DONE"
