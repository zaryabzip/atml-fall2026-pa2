#!/bin/zsh

cd "${0:A:h}/.."
PY=.venv/bin/python
CFG=configs/feedback.yaml
export PYTHONUNBUFFERED=1

log() { echo "$(date '+%F %T') $*"; }

run_stage() {
  local name=$1; shift
  log "START $name: $*"
  "$@"
  local code=$?
  if (( code != 0 )); then
    log "FAILED $name (exit $code). Pipeline stopped. Rerun scripts/run_task5.sh to continue."
    exit $code
  fi
  log "DONE $name"
}

while pgrep -f "scripts/run_task[1234].sh" >/dev/null || pgrep -f "_after.sh" >/dev/null || pgrep -f "m task[1234]_[a-z]*[.]" >/dev/null; do
  log "waiting for other task jobs to finish"
  sleep 300
done

run_stage "gsm eval" $PY -m task5_feedback.evaluate_math --config $CFG --dataset gsm
run_stage "diagnostic scoring" $PY -m task5_feedback.score_perturbations --config $CFG
run_stage "transfer eval" $PY -m task5_feedback.evaluate_math --config $CFG --dataset transfer
run_stage "feedback comparison" $PY -m task5_feedback.compare_feedback --config $CFG

log "ALL TASK 5 RUNS DONE"
