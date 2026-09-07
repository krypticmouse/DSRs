#!/bin/bash
# HARVEST ablation lane — Table A of docs/harvest-ablation-prereg.md.
# Serial DocsQA search arms on toast (ALONE on that provider), every arm
# from the frozen v6.3 campaign binary (bin/36-docsqa-v6.3-campaign, the
# byte-identical binary that produced Tier-3 look #2). One knob changes per
# arm; everything else is look #2's environment (run_harvest5_chain.sh).
# The verdict is NOT run here: it is blocked on holdout_b being minted and
# sealed (prereg §2/§7). Zero-rate guard: > 6% score-0 rollouts voids an arm.
set -u
MAIN=/Users/krypticmouse/Desktop/DSRs
D=$MAIN/benchmarks/docsqa
BIN=$D/bin/36-docsqa-v6.3-campaign
LOG=$D/ablation_lane.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
set -a; source "$MAIN/.env"; set +a
cd "$MAIN"

probe_toast() {
  curl -s -m 90 -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $MXBAI_API_KEY" -H "Content-Type: application/json" \
    https://api.mixedbread.com/v1/chat/completions \
    -d '{"model":"toast-1","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":16}' \
    || echo 000
}

preflight() {
  for i in 1 2 3 4 5 6; do
    local T O
    T=$(probe_toast)
    O=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://openrouter.ai/api/v1/models || echo 000)
    log "pre-flight $i: toast $T openrouter $O"
    [ "$T" = 200 ] && [ "$O" = 200 ] && return 0
    sleep 300
  done
  return 1
}

# Toast must be ours alone: no other DocsQA optimizer binary or pip-gepa run.
# (Match binary paths, never the bare name — a watcher shell whose command
# text contains "36-docsqa" stalled an earlier chain for an hour.)
wait_toast_idle() {
  while pgrep -f 'examples/36-docsqa|36-docsqa-v6|run_python_gepa.py' >/dev/null; do sleep 120; done
}

COMMON="DOCSQA_DIR=$D DOCSQA_MODE=optimize DOCSQA_OPT=harvest DOCSQA_ALLOW_MEM=1 \
  DOCSQA_CHAMPION=30 DOCSQA_CHAMPION_K=4 DOCSQA_CERTIFY_RESERVE=0 \
  DOCSQA_CERTIFY_FILE=pool_certify.jsonl DOCSQA_CERTIFY_BUDGET=800 \
  DOCSQA_CERTIFY_BAND=horizon DOCSQA_CONCURRENCY=8 DOCSQA_BUDGET=3100"

run_arm() {  # tag seed [KEY=VAL ...]
  local TAG=$1 SEED=$2; shift 2
  local RUNLOG="$D/${TAG}_s${SEED}.log"
  wait_toast_idle
  preflight || { log "$TAG: providers down after 30 min; lane aborted"; exit 1; }
  log "$TAG (seed $SEED): start [${*:-preset}]"
  env $COMMON DOCSQA_SEED=$SEED DOCSQA_OUT_PREFIX=$TAG "$@" "$BIN" > "$RUNLOG" 2>&1
  local RC=$?
  local RL TOT ZEROS PCT
  RL=$(grep -a -E '^rollouts:' "$RUNLOG" | tail -1)
  TOT=$(echo "$RL" | sed -E 's/^rollouts: ([0-9]+).*/\1/')
  ZEROS=$(grep -a -c 'scored 0' "$RUNLOG")
  PCT=$(python3 -c "print(round(100*${ZEROS:-0}/max(1,${TOT:-0}),1))")
  log "$TAG: exit $RC :: $(grep -a '^certification:' "$RUNLOG" | tail -1) :: ${RL:-no rollouts line} :: zeros $ZEROS (${PCT}%) :: $(grep -a '^screen ' "$RUNLOG" | tail -1)"
  if python3 -c "import sys; sys.exit(0 if ${PCT} > 6 else 1)"; then
    log "$TAG: VOID (zero-rate ${PCT}% > 6%) — winner files renamed *.void.json; rerun alone before any reading"
    for f in "$D/${TAG}_s${SEED}.json" "$D/${TAG}_argmax_s${SEED}.json"; do
      [ -f "$f" ] && mv "$f" "${f%.json}.void.json"
    done
  fi
}

log "LANE START (binary $(shasum -a 256 "$BIN" | cut -c1-12)…)"
if [ -s "$D/harvest_full2_s1.json" ]; then
  log "A0-s1 (harvest_full2 seed 1) already in hand (certified 2026-09-02 16:15); skipping"
else
  run_arm harvest_full2 1                                        # A0-s1: second search seed, preset
fi
run_arm abl_nopolicy  0 DOCSQA_POLICY=0 DOCSQA_POLICY_SHARE=0    # A1: no Policy operator / channel
run_arm abl_screenb   0 DOCSQA_SCREEN=b DOCSQA_GATE_SIZE=5       # A2: held-out batch-B screen
run_arm abl_scout16   0 DOCSQA_ACCEPT=gate DOCSQA_SCOUT=16       # A3: 16-column scout, not full slice
run_arm abl_allops    0 DOCSQA_REWRITE_ONLY=0                    # A4: all four operators back
run_arm abl_greed5    0 DOCSQA_PARENT_GREED=0.5                  # A5: parent greed .5

# G-s1: pip-gepa at 2000 metric calls, second seed (same student/judge via serve)
wait_toast_idle
preflight || { log "gepa b2000 s1: providers down; lane aborted"; exit 1; }
log "gepa b2000 seed 1: start"
(cd "$D" && DOCSQA_SEED=1 DOCSQA_BUDGET=2000 DOCSQA_TAG=b2000 \
  ../bcplus/.venv/bin/python run_python_gepa.py > gepa_b2000_s1.log 2>&1)
log "gepa b2000 s1: exit $? :: $(ls -la "$D/python_winner_b2000_s1.json" 2>/dev/null | awk '{print $5" bytes"}')"
log "LANE DONE — Table A verdict pending holdout_b seal (docs/harvest-ablation-prereg.md §3)"
