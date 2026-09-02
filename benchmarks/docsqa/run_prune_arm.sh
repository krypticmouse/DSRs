#!/bin/bash
# Lever L-A (docs/harvest-ablation-prereg.md §6): HARVEST v6.3 preset +
# post-certification Occam pruning ordered by lineage-diff clause credit
# (`prune_after_certify`), full DocsQA budget so the size problem it targets
# (8.5–11.5k winners) is actually on the table. Runs from the frozen v6.4
# binary (bin/36-docsqa-v6.4-prune). 2026-09-02: the paper-wide ablation
# lane was stopped by the user; this script now waits only for the one arm
# still running (harvest_full2 seed 1, v6.3 campaign binary), books its
# summary line, then runs the pruning arm. No holdout look here; the
# certification + chars before/after on pool_certify are the readout.
set -u
MAIN=/Users/krypticmouse/Desktop/DSRs
D=$MAIN/benchmarks/docsqa
BIN=$D/bin/36-docsqa-v6.4-prune
LOG=$D/ablation_lane.log
log() { echo "$(date '+%F %T') [prune] $*" >> "$LOG"; }
set -a; source "$MAIN/.env"; set +a
cd "$MAIN"

summarize() {  # tag seed
  local RUNLOG="$D/$1_s$2.log" RL TOT ZEROS PCT
  RL=$(grep -a -E '^rollouts:' "$RUNLOG" | tail -1)
  TOT=$(echo "$RL" | sed -E 's/^rollouts: ([0-9]+).*/\1/')
  ZEROS=$(grep -a -c 'scored 0' "$RUNLOG")
  PCT=$(python3 -c "print(round(100*${ZEROS:-0}/max(1,${TOT:-0}),1))")
  log "$1 (seed $2): $(grep -a '^certification:' "$RUNLOG" | tail -1) :: ${RL:-no rollouts line} :: zeros $ZEROS (${PCT}%) :: winner $(wc -c < "$D/$1_s$2.json" 2>/dev/null) chars"
  if python3 -c "import sys; sys.exit(0 if ${PCT} > 6 else 1)"; then
    log "$1: VOID (zero-rate ${PCT}% > 6%) — winner files renamed *.void.json"
    for f in "$D/$1_s$2.json" "$D/$1_argmax_s$2.json"; do [ -f "$f" ] && mv "$f" "${f%.json}.void.json"; done
  fi
}

log "waiting for harvest_full2 seed 1 (the last ablation arm left running) to exit"
while pgrep -f '36-docsqa-v6.3-campaign' >/dev/null; do sleep 120; done
summarize harvest_full2 1
while pgrep -f 'examples/36-docsqa|36-docsqa-v6|run_python_gepa.py' >/dev/null; do sleep 120; done

for i in 1 2 3 4 5 6; do
  T=$(curl -s -m 90 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $MXBAI_API_KEY" -H "Content-Type: application/json" \
      https://api.mixedbread.com/v1/chat/completions \
      -d '{"model":"toast-1","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":16}' || echo 000)
  log "pre-flight $i: toast $T"
  [ "$T" = 200 ] && break
  [ "$i" = 6 ] && { log "toast down after 30 min; not started"; exit 1; }
  sleep 300
done

COMMON="DOCSQA_DIR=$D DOCSQA_MODE=optimize DOCSQA_OPT=harvest DOCSQA_ALLOW_MEM=1 \
  DOCSQA_CHAMPION=30 DOCSQA_CHAMPION_K=4 DOCSQA_CERTIFY_RESERVE=0 \
  DOCSQA_CERTIFY_FILE=pool_certify.jsonl DOCSQA_CERTIFY_BUDGET=800 \
  DOCSQA_CERTIFY_BAND=horizon DOCSQA_CONCURRENCY=8"
TAG=abl_prune; RUNLOG="$D/${TAG}_s0.log"
log "$TAG (seed 0): start [DOCSQA_PRUNE=1 DOCSQA_PRUNE_BUDGET=120, budget 3220] binary $(shasum -a 256 "$BIN" | cut -c1-12)…"
env $COMMON DOCSQA_SEED=0 DOCSQA_OUT_PREFIX=$TAG DOCSQA_BUDGET=3220 DOCSQA_PRUNE=1 DOCSQA_PRUNE_BUDGET=120 \
  "$BIN" > "$RUNLOG" 2>&1
log "$TAG: exit $?"
summarize $TAG 0
log "$TAG: prune report :: $(grep -a -i -E 'prune' "$RUNLOG" | grep -a -v DEBUG | tail -n 4 | tr '\n' '|')"
log "PRUNE ARM DONE"
