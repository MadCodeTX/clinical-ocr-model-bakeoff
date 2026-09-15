#!/usr/bin/env bash
# Run the dots family with their native layout prompt: JSON of ordered regions,
# each with bbox + category + text (tables as HTML). Tests whether this gets us
# bboxes AND selection controls together, which no plain-prompt model managed.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MAX_LEN=16384
export MAX_TOKENS=8192
LOG=logs/layout_prompt.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

cleanup() { docker rm -f ocr-layout-dots-mocr ocr-layout-dots-ocr >/dev/null 2>&1 || true; }
cleanup

PROMPT="$(cat prompts/dots_layout.txt)"
run_one() {  # run_one NAME HF_ID GPU
  local name=$1 hf=$2 gpu=$3
  if [ -f "results/$name/summary.json" ]; then log "skip $name"; return 0; fi
  log "layout-prompt eval $name (GPU$gpu)"
  PROMPT="$PROMPT" bash serve_and_eval.sh "$name" "$hf" "$gpu" data/clinocr zero 6 \
    > "logs/eval_$name.log" 2>&1 || log "FAIL $name"
  .venv/bin/python3 score_layout.py --name "$name" --data data/clinocr >> "$LOG" 2>&1 || log "score FAIL $name"
}

run_one layout-dots-mocr rednote-hilab/dots.mocr 0 &
A=$!
run_one layout-dots-ocr  rednote-hilab/dots.ocr  1 &
B=$!
wait $A $B
cleanup
log "layout-prompt test complete"
echo "done $(date -u '+%Y-%m-%d %H:%M UTC')" > logs/layout_prompt.done
