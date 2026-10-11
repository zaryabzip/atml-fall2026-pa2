#!/bin/zsh

cd "${0:A:h}/.."
PY=.venv/bin/python
CFG=configs/grpo.yaml
export PYTHONUNBUFFERED=1

log() { echo "$(date '+%F %T') $*"; }

run_stage() {
  local name=$1; shift
  log "START $name: $*"
  "$@"
  local code=$?
  if (( code != 0 )); then
    log "FAILED $name (exit $code). Pipeline stopped. Rerun scripts/run_task3.sh to continue."
    exit $code
  fi
  log "DONE $name"
}

while pgrep -f "scripts/run_task[12].sh" >/dev/null || pgrep -f "_after.sh" >/dev/null || pgrep -f "m task[123]_[a-z]*[.]" >/dev/null; do
  log "waiting for other task jobs to finish"
  sleep 300
done

if [[ -f outputs/task3_grpo/standard/adapter_config.json ]]; then
  log "SKIP standard grpo training (adapter exists)"
else
  run_stage "train standard grpo" $PY -m task3_grpo.continue_train --config $CFG --run-name standard --resume
fi

if [[ -f results/task3_grpo/midpoint_eval.json ]]; then
  log "SKIP grpo midpoint evaluation (results exist)"
else
  run_stage "evaluate grpo midpoint" $PY -m task3_grpo.evaluate --config $CFG --adapter checkpoints/grpo_midpoint_policy --name midpoint
fi

if [[ -f results/task3_grpo/standard_eval.json ]]; then
  log "SKIP standard grpo evaluation (results exist)"
else
  run_stage "evaluate standard grpo" $PY -m task3_grpo.evaluate --config $CFG --adapter outputs/task3_grpo/standard --name standard
fi

if [[ -f results/task3_grpo/group_size_study.json ]]; then
  log "SKIP group size study (results exist)"
else
  run_stage "group size study" $PY -m task3_grpo.analyze_group_size --config $CFG
fi

if [[ -f results/task3_grpo/normalization_study.json ]]; then
  log "SKIP normalization study (results exist)"
else
  run_stage "normalization study" $PY -m task3_grpo.compare_normalization --config $CFG
fi

log "ALL TASK 3 RUNS DONE"
