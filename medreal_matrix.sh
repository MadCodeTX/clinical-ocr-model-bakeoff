#!/usr/bin/env bash
# Run every model in the bake-off on the NEW medreal corpus, so we can compare
# each model's behaviour on the old ClinOCR-Bench vs the new real-scan set.
#
# Two single-GPU lanes run concurrently; the 27B (TP=2) runs after they free
# both cards; Tesseract runs on CPU throughout. Resumable: a run whose
# results/medreal-<name>/summary.json exists is skipped.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
LOG=logs/medreal_matrix.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

cleanup() { docker ps --format '{{.Names}}' | grep '^ocr-' | xargs -r docker rm -f >/dev/null 2>&1 || true; }

run_list() {  # run_list GPU  (name hf_id prompt)...
  local gpu=$1; shift
  while [ $# -ge 3 ]; do
    local name=$1 hf=$2 prompt=$3; shift 3
    if [ -f "results/medreal-$name/summary.json" ]; then log "skip medreal-$name"; continue; fi
    log "eval medreal-$name on GPU$gpu"
    PROMPT="$prompt" bash serve_and_eval.sh "medreal-$name" "$hf" "$gpu" data/medreal zero 8 \
      > "logs/eval_medreal-$name.log" 2>&1 || log "FAIL medreal-$name (see logs/eval_medreal-$name.log)"
  done
}

cleanup
run_list 0 \
  dots-mocr      rednote-hilab/dots.mocr            "Extract the text content from this image." \
  granite-docling ibm-granite/granite-docling-258M  "Convert this page to docling." \
  dots-ocr       rednote-hilab/dots.ocr             "Extract the text content from this image." \
  olmocr-2       allenai/olmOCR-2-7B-1025           "Extract the contents. [Markdown]." \
  deepseek-ocr   deepseek-ai/DeepSeek-OCR           "Free OCR." \
  chandra-2      datalab-to/chandra-ocr-2           "Convert this page to markdown." &
P0=$!
run_list 1 \
  paddleocr-vl   PaddlePaddle/PaddleOCR-VL          "OCR:" \
  qwen25vl3b-base Qwen/Qwen2.5-VL-3B-Instruct       "Extract the text content from this image." \
  qwen25vl7b     Qwen/Qwen2.5-VL-7B-Instruct        "Extract the text content from this image." \
  nanonets-ocr2-3b nanonets/Nanonets-OCR2-3B        "Convert this document to markdown." &
P1=$!

# Tesseract on CPU (no GPU needed) at the same time
( DATA=data/medreal OUT=results/medreal-tesseract bash eval_tesseract.sh \
    > logs/eval_medreal-tesseract.log 2>&1 ) &
PT=$!

wait $P0 $P1
cleanup

# 27B: tensor-parallel across both cards, so it runs alone
if [ ! -f results/medreal-qwen38-27b/summary.json ]; then
  log "eval medreal-qwen38-27b (TP=2, both GPUs)"
  PROMPT="Extract the text content from this image." \
    bash serve_and_eval.sh medreal-qwen38-27b Qwen/Qwen3.8-27B-FP8 0,1 data/medreal zero 8 \
    > logs/eval_medreal-qwen38-27b.log 2>&1 || log "FAIL medreal-qwen38-27b"
fi
cleanup

# old-only student on the new set, for the training comparison
if [ -d checkpoints/q38c-7b-800 ] && [ ! -f results/medreal-student7b-oldonly/summary.json ]; then
  log "eval medreal-student7b-oldonly (GPU0)"
  LORA_TOWER=1 MAX_PIXELS=802816 DATA=data/medreal \
    bash run_lora_model.sh medreal-student7b-oldonly checkpoints/q38c-7b-800 0 \
    Qwen/Qwen2.5-VL-7B-Instruct > logs/eval_medreal-student7b-oldonly.log 2>&1 \
    || log "FAIL medreal-student7b-oldonly"
fi
wait $PT 2>/dev/null || true
cleanup

log "medreal matrix complete"
.venv/bin/python3 make_overnight_reports.py >> "$LOG" 2>&1 || true
echo "done $(date -u '+%Y-%m-%d %H:%M UTC')" > logs/medreal_matrix.done
