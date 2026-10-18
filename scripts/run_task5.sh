#!/bin/zsh

cd "${0:A:h}/.."  # run from the repo root
PY=.venv/bin/python  # venv python
CFG=configs/feedback.yaml  # task config
export PYTHONUNBUFFERED=1  # print logs right away

log() { echo "$(date '+%F %T') $*"; }  # timestamped log line

run_stage() {  # run one stage and stop the pipeline if it fails
  local name=$1; shift
  log "START $name: $*"
  "$@"  # run the stage command
  local code=$?
  if (( code != 0 )); then  # stage failed
    log "FAILED $name (exit $code). Pipeline stopped. Rerun scripts/run_task5.sh to continue."
    exit $code
  fi
  log "DONE $name"
}

while pgrep -f "scripts/run_task[1234].sh" >/dev/null || pgrep -f "_after.sh" >/dev/null || pgrep -f "m task[1234]_[a-z]*[.]" >/dev/null; do  # wait until no other job is running
  log "waiting for other task jobs to finish"
  sleep 300
done

run_stage "gsm eval" $PY -m task5_feedback.evaluate_math --config $CFG --dataset gsm  # gsm8k accuracy and judge win rates
run_stage "diagnostic scoring" $PY -m task5_feedback.score_perturbations --config $CFG  # verifier and judge on perturbed answers
run_stage "transfer eval" $PY -m task5_feedback.evaluate_math --config $CFG --dataset transfer  # svamp transfer
run_stage "feedback comparison" $PY -m task5_feedback.compare_feedback --config $CFG  # final rlvr vs rlaif tables

log "ALL TASK 5 RUNS DONE"
