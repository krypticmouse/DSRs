#!/bin/bash
# HARVEST ablation lane — Table B of docs/harvest-ablation-prereg.md.
# PUPA / gpt-oss-20b, budget 600, one knob changed per arm, from a frozen
# campaign binary (bin/33-pupa-campaign, built from the v6 worktree after
# the 2026-09-02 driver knobs; sha256 recorded in the prereg ledger).
# Waits for the BrowseComp+ oss120 cell to release openrouter (one student
# arm per provider at a time). The sealed holdout100 verdict is NOT run
# here — the lane stops at READY FOR VERDICT and prints the command.
set -u
MAIN=/Users/krypticmouse/Desktop/DSRs
D=$MAIN/benchmarks/pupa
BIN=$D/bin/33-pupa-campaign
LOG=$D/ablation_lane.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
set -a; source "$MAIN/.env"; set +a
cd "$MAIN"

[ -x "$BIN" ] || { log "no frozen binary at $BIN; lane not started"; exit 1; }

log "queued: waiting for the bcplus oss120 cell (run_harvest_cell_120b.sh) to finish"
while pgrep -f "run_harvest_cell_120b.sh" >/dev/null; do sleep 300; done
# and for any other openrouter student arm (pupa/hover/bcplus binaries, pip-gepa)
wait_openrouter_idle() {
  while pgrep -f 'examples/31-bcplus|examples/33-pupa|examples/34-hover|33-pupa-campaign|run_python_gepa.py' >/dev/null; do sleep 120; done
}
preflight() {
  for i in 1 2 3 4 5 6; do
    local O
    O=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://openrouter.ai/api/v1/models || echo 000)
    log "pre-flight $i: openrouter $O"
    [ "$O" = 200 ] && return 0
    sleep 300
  done
  return 1
}

# Grid-cell environment (benchmarks/run_harvest_grid.sh run_cell, pupa/oss20).
COMMON="PUPA_STUDENT=openai/gpt-oss-20b PUPA_TEACHER=openrouter:openai/gpt-5.6-terra \
  PUPA_BUDGET=600 PUPA_VAL_FILE=val.jsonl PUPA_CONCURRENCY=6 PUPA_DIR=benchmarks/pupa \
  PUPA_MODE=optimize PUPA_OPT=harvest PUPA_CERTIFY_RESERVE=45 PUPA_CERTIFY_BUDGET=150"

run_arm() {  # tag seed [KEY=VAL ...]
  local TAG=$1 SEED=$2; shift 2
  local RUNLOG="$D/${TAG}_run_s${SEED}.log"
  wait_openrouter_idle
  preflight || { log "$TAG: openrouter down after 30 min; lane aborted"; exit 1; }
  log "$TAG (seed $SEED): start [${*:-preset}]"
  env $COMMON PUPA_SEED=$SEED PUPA_OUT_PREFIX="${TAG}_winner" "$@" "$BIN" > "$RUNLOG" 2>&1
  local RC=$?
  local ERRS
  ERRS=$(grep -a -c -E 'ROLLOUT ERROR|scored 0|call failed' "$RUNLOG")
  log "$TAG: exit $RC :: $(grep -a '^certification:' "$RUNLOG" | tail -1) :: $(grep -a -E '^rollouts:' "$RUNLOG" | tail -1) :: error-ish lines $ERRS :: $(grep -a '^harvest:' "$RUNLOG" | tail -1)"
}

log "LANE START (binary $(shasum -a 256 "$BIN" | cut -c1-12)…)"
run_arm oss20_harvest      1                                     # B0-s1: second search seed, preset
run_arm oss20_abl_nopolicy 0 PUPA_POLICY=0 PUPA_POLICY_SHARE=0   # B1: no Policy operator / channel
run_arm oss20_abl_screenb  0 PUPA_SCREEN=b PUPA_GATE_SIZE=5      # B2: held-out batch-B screen
run_arm oss20_abl_scout16  0 PUPA_SCOUT=16                       # B3: 16-column scout
run_arm oss20_abl_allops   0 PUPA_REWRITE_ONLY=0                 # B4: all four operators back

# pygepa second seed (pip-gepa, same student/judge via serve)
wait_openrouter_idle
preflight || { log "pygepa s1: openrouter down; lane aborted"; exit 1; }
log "pygepa oss20 seed 1: start"
(cd "$D" && PUPA_STUDENT=openai/gpt-oss-20b PUPA_TAG=oss20 PUPA_SEED=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python run_python_gepa.py > oss20_pygepa_run_s1.log 2>&1)
log "pygepa s1: exit $? :: $(ls -la "$D/python_winner_oss20_s1.txt" 2>/dev/null | awk '{print $5" bytes"}')"

MISSING=""
for f in oss20_harvest_winner_s0.txt oss20_harvest_winner_s1.txt oss20_abl_nopolicy_winner_s0.txt \
         oss20_abl_screenb_winner_s0.txt oss20_abl_scout16_winner_s0.txt oss20_abl_allops_winner_s0.txt \
         oss20_ratchet_winner_s0.txt oss20_mergedgepa_winner_s0.txt python_winner_oss20_s0.txt python_winner_oss20_s1.txt; do
  [ -s "$D/$f" ] || MISSING="$MISSING $f"
done
log "LANE DONE — READY FOR VERDICT (sealed holdout100 x2, ONE look; file alpha ledger row first). Missing:${MISSING:- none}"
log "verdict cmd: cd $D && PUPA_STUDENT=openai/gpt-oss-20b PYTHONUNBUFFERED=1 .venv/bin/python eval_holdout.py --split holdout100 --reps 2 --out eval_oss20_ablation_holdout100.jsonl seed=SEED harvest=oss20_harvest_winner_s0.txt harvest_s1=oss20_harvest_winner_s1.txt nopolicy=oss20_abl_nopolicy_winner_s0.txt screenb=oss20_abl_screenb_winner_s0.txt scout16=oss20_abl_scout16_winner_s0.txt allops=oss20_abl_allops_winner_s0.txt ratchet=oss20_ratchet_winner_s0.txt mergedgepa=oss20_mergedgepa_winner_s0.txt pygepa=python_winner_oss20_s0.txt pygepa_s1=python_winner_oss20_s1.txt > oss20_ablation_verdict.log 2>&1"
