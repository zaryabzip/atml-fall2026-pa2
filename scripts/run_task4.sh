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
    log "FAILED $name (exit $code). Pipeline stopped. Rerun scripts/run_task4.sh to continue."
    exit $code
  fi
  log "DONE $name"
}

while pgrep -f "scripts/run_task[123].sh" >/dev/null || pgrep -f "_after.sh" >/dev/null || pgrep -f "m task[123]_[a-z]*[.]" >/dev/null; do
  log "waiting for other task jobs to finish"
  sleep 300
done

run_stage "generate responses" $PY -m task4_safety.generate_responses --config $CFG
run_stage "audit sheet" $PY -m task4_safety.make_audit_sheet --config $CFG
run_stage "judge responses" $PY -m task4_safety.judge_responses --config $CFG
run_stage "safety summary" $PY -m task4_safety.evaluate_safety --config $CFG

log "ALL TASK 4 RUNS DONE"
