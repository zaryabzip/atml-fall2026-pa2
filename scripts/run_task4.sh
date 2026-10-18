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
    log "FAILED $name (exit $code). Pipeline stopped. Rerun scripts/run_task4.sh to continue."
    exit $code
  fi
  log "DONE $name"
}

while pgrep -f "scripts/run_task[123].sh" >/dev/null || pgrep -f "_after.sh" >/dev/null || pgrep -f "m task[123]_[a-z]*[.]" >/dev/null; do  # wait until no other job is running
  log "waiting for other task jobs to finish"
  sleep 300
done

run_stage "generate responses" $PY -m task4_safety.generate_responses --config $CFG  # greedy xstest answers for each policy
run_stage "audit sheet" $PY -m task4_safety.make_audit_sheet --config $CFG  # blind sheet for the manual audit
run_stage "judge responses" $PY -m task4_safety.judge_responses --config $CFG  # ai judge labels
run_stage "safety summary" $PY -m task4_safety.evaluate_safety --config $CFG  # rates and audit agreement

log "ALL TASK 4 RUNS DONE"
