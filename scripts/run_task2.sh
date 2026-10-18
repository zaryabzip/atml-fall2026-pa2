#!/bin/zsh

cd "${0:A:h}/.."  # run from the repo root
PY=.venv/bin/python  # venv python
CFG=configs/ppo.yaml  # task config
export PYTHONUNBUFFERED=1  # print logs right away

log() { echo "$(date '+%F %T') $*"; }  # timestamped log line

run_stage() {  # run one stage and stop the pipeline if it fails
  local name=$1; shift
  log "START $name: $*"
  "$@"  # run the stage command
  local code=$?
  if (( code != 0 )); then  # stage failed
    log "FAILED $name (exit $code). Pipeline stopped. Rerun scripts/run_task2.sh to continue."
    exit $code
  fi
  log "DONE $name"
}

while pgrep -f "scripts/run_task1.sh" >/dev/null || pgrep -f "base_eval_after" >/dev/null || pgrep -f "m task[12]_[a-z]*[.]" >/dev/null; do  # wait until no other job is running
  log "waiting for task 1 jobs to finish"
  sleep 300
done

if [[ -f outputs/task2_ppo/standard/adapter_config.json ]]; then  # skip training if the adapter is saved
  log "SKIP standard ppo training (adapter exists)"
else
  run_stage "train standard ppo" $PY -m task2_ppo.continue_train --config $CFG --run-name standard --resume  # resumes from the last checkpoint
fi

if [[ -f results/task2_ppo/midpoint_eval.json ]]; then  # skip if results are saved
  log "SKIP midpoint evaluation (results exist)"
else
  run_stage "evaluate ppo midpoint" $PY -m task2_ppo.evaluate --config $CFG --adapter checkpoints/ppo_midpoint_policy --name midpoint
fi

if [[ -f results/task2_ppo/standard_eval.json ]]; then  # skip if results are saved
  log "SKIP standard ppo evaluation (results exist)"
else
  run_stage "evaluate standard ppo" $PY -m task2_ppo.evaluate --config $CFG --adapter outputs/task2_ppo/standard --name standard
fi

if [[ -f results/task2_ppo/clipping_study.json ]]; then  # skip if results are saved
  log "SKIP clipping study (results exist)"
else
  run_stage "clipping study" $PY -m task2_ppo.analyze_clipping --config $CFG
fi

if [[ -f results/task2_ppo/kl_study.json ]]; then  # skip if results are saved
  log "SKIP kl study (results exist)"
else
  run_stage "kl study" $PY -m task2_ppo.ablate_kl --config $CFG
fi

log "ALL TASK 2 RUNS DONE"
