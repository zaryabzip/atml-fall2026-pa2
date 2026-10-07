#!/bin/zsh

cd "${0:A:h}/.."
PY=.venv/bin/python
CFG=configs/dpo.yaml
export PYTHONUNBUFFERED=1

log() { echo "$(date '+%F %T') $*"; }

run_stage() {
  local name=$1; shift
  log "START $name: $*"
  "$@"
  local code=$?
  if (( code != 0 )); then
    log "FAILED $name (exit $code). Pipeline stopped. Rerun scripts/run_task1.sh to continue."
    exit $code
  fi
  log "DONE $name"
}

while pgrep -f "task1_dpo\.(train|evaluate|ablate_beta|analyze_length)" >/dev/null; do
  log "waiting for a running Task 1 job to finish"
  sleep 300
done

if [[ -f outputs/task1_dpo/standard/adapter_config.json ]]; then
  log "SKIP standard training (adapter exists)"
else
  run_stage "train standard" $PY -m task1_dpo.train --config $CFG --run-name standard --resume
fi

if [[ -f results/task1_dpo/standard_eval.json ]]; then
  log "SKIP standard evaluation (results exist)"
else
  run_stage "evaluate standard" $PY -m task1_dpo.evaluate --config $CFG --adapter outputs/task1_dpo/standard --name standard
fi

if [[ -f results/task1_dpo/beta_ablation.json ]]; then
  log "SKIP beta ablation (results exist)"
else
  run_stage "beta ablation" $PY -m task1_dpo.ablate_beta --config $CFG
fi

if [[ -f results/task1_dpo/length_analysis.json ]]; then
  log "SKIP length analysis (results exist)"
else
  run_stage "length analysis" $PY -m task1_dpo.analyze_length --config $CFG
fi

log "ALL TASK 1 RUNS DONE"
