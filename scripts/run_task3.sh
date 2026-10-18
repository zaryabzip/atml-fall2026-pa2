#!/bin/zsh

cd "${0:A:h}/.."  # run from the repo root
PY=.venv/bin/python  # venv python
CFG=configs/grpo.yaml  # task config
export PYTHONUNBUFFERED=1  # print logs right away

log() { echo "$(date '+%F %T') $*"; }  # timestamped log line

run_stage() {  # run one stage and stop the pipeline if it fails
  local name=$1; shift
  log "START $name: $*"
  "$@"  # run the stage command
  local code=$?
  if (( code != 0 )); then  # stage failed
    log "FAILED $name (exit $code). Pipeline stopped. Rerun scripts/run_task3.sh to continue."
    exit $code
  fi
  log "DONE $name"
}

while pgrep -f "scripts/run_task[12].sh" >/dev/null || pgrep -f "_after.sh" >/dev/null || pgrep -f "m task[123]_[a-z]*[.]" >/dev/null; do  # wait until no other job is running
  log "waiting for other task jobs to finish"
  sleep 300
done

if [[ -f outputs/task3_grpo/standard/adapter_config.json ]]; then  # skip training if the adapter is saved
  log "SKIP standard grpo training (adapter exists)"
else
  run_stage "train standard grpo" $PY -m task3_grpo.continue_train --config $CFG --run-name standard --resume  # resumes from the last checkpoint
fi

if [[ -f results/task3_grpo/midpoint_eval.json ]]; then  # skip if results are saved
  log "SKIP grpo midpoint evaluation (results exist)"
else
  run_stage "evaluate grpo midpoint" $PY -m task3_grpo.evaluate --config $CFG --adapter checkpoints/grpo_midpoint_policy --name midpoint
fi

if [[ -f results/task3_grpo/standard_eval.json ]]; then  # skip if results are saved
  log "SKIP standard grpo evaluation (results exist)"
else
  run_stage "evaluate standard grpo" $PY -m task3_grpo.evaluate --config $CFG --adapter outputs/task3_grpo/standard --name standard
fi

if [[ -f results/task3_grpo/group_size_study.json ]]; then  # skip if results are saved
  log "SKIP group size study (results exist)"
else
  run_stage "group size study" $PY -m task3_grpo.analyze_group_size --config $CFG
fi

if [[ -f results/task3_grpo/normalization_study.json ]]; then  # skip if results are saved
  log "SKIP normalization study (results exist)"
else
  run_stage "normalization study" $PY -m task3_grpo.compare_normalization --config $CFG
fi

log "ALL TASK 3 RUNS DONE"
