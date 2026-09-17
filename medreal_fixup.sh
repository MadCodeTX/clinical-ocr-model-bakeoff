#!/usr/bin/env bash
# Re-run the three medreal evaluations that were corrupted: dots.mocr (killed
# mid-eval), the old-only student (killed), and the 27B (connection resets).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
LOG=logs/medreal_fixup.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
clean() { docker rm -f ocr-medreal-dots-mocr ocr-lora-medreal-student7b-oldonly ocr-medreal-qwen38-27b >/dev/null 2>&1 || true; }

clean
rm -rf results/medreal-dots-mocr results/medreal-qwen38-27b results/medreal-student7b-oldonly

( PROMPT="Extract the text content from this image." bash serve_and_eval.sh \
    medreal-dots-mocr rednote-hilab/dots.mocr 0 data/medreal zero 8 \
    > logs/eval_medreal-dots-mocr.log 2>&1; log "dots.mocr done" ) &
A=$!
( LORA_TOWER=1 MAX_PIXELS=802816 DATA=data/medreal bash run_lora_model.sh \
    medreal-student7b-oldonly checkpoints/q38c-7b-800 1 Qwen/Qwen2.5-VL-7B-Instruct \
    > logs/eval_medreal-student7b-oldonly.log 2>&1; log "oldonly done" ) &
B=$!
wait $A $B
clean

log "27B medreal re-run (TP=2, larger context, c=6)"
PROMPT="Extract the text content from this image." MAX_LEN=16384 \
  bash serve_and_eval.sh medreal-qwen38-27b Qwen/Qwen3.8-27B-FP8 0,1 data/medreal zero 6 \
  > logs/eval_medreal-qwen38-27b.log 2>&1 || log "27B FAILED"
clean
.venv/bin/python3 make_overnight_reports.py >> "$LOG" 2>&1 || true
log "medreal fixup complete"
echo "done $(date -u '+%Y-%m-%d %H:%M UTC')" > logs/medreal_fixup.done
