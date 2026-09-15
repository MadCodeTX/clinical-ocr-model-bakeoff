#!/usr/bin/env bash
# Re-run the corpus-size sweep on the CLEANED teacher corpus (reasoning preamble
# stripped by build_teacher_set.py). The earlier sweep adapters were trained on
# transcripts that still carried Qwen3.8's  thinking preamble, so they learned to
# emit it and scored worse than the base model; those results are discarded here.
#
# Training runs on GPU1; each finished adapter is evaluated on GPU0 while the
# next model trains, so the two cards stay busy and evals stay serialised.
#
# Usage: bash rerun_sweep.sh   (nohup/setsid for unattended)
set -uo pipefail
cd "$(dirname "$0")" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/rerun_sweep.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

LABELS=data/synth_large/teacher_labels_q38.jsonl
B3=Qwen/Qwen2.5-VL-3B-Instruct
B7=Qwen/Qwen2.5-VL-7B-Instruct
TRAIN_GPU=1
EVAL_GPU=0
EVAL_PID=""

cleanup() { docker ps --format '{{.Names}}' | grep '^ocr-' | xargs -r docker rm -f >/dev/null 2>&1 || true; }

stop_eval() {
  if [ -n "$EVAL_PID" ]; then
    wait "$EVAL_PID" 2>/dev/null || true
    EVAL_PID=""
  fi
}

run_one() {  # run_one <base> <n> <tag> <ckpt>
  local base=$1 n=$2 tag=$3 ckpt=$4
  if [ -d "checkpoints/$ckpt" ] && [ -f "results/$tag-$n/summary.json" ]; then
    log "SKIP $tag-$n (adapter + result already present)"; return 0
  fi
  head -n "$n" "$LABELS" > "/tmp/clean_${tag}_$n.jsonl"
  if [ ! -d "checkpoints/$ckpt" ]; then
    log "TRAIN $tag-$n  ($base, $n docs) -> checkpoints/$ckpt"
    if ! CUDA_VISIBLE_DEVICES=$TRAIN_GPU train-venv/bin/python3 train_student.py \
        --model "$base" --data "/tmp/clean_${tag}_$n.jsonl" \
        --out "checkpoints/$ckpt" --epochs 1 > "logs/train_${ckpt}.log" 2>&1; then
      log "TRAIN FAILED $ckpt (see logs/train_${ckpt}.log)"; return 1
    fi
  fi
  if [ -f "results/$tag-$n/summary.json" ]; then
    log "SKIP eval $tag-$n (already evaluated)"; return 0
  fi
  # make sure the previous eval released GPU0 before starting the next one
  stop_eval
  log "EVAL  $tag-$n  (GPU$EVAL_GPU)"
  (
    LORA_TOWER=1 MAX_PIXELS=802816 bash run_lora_model.sh "$tag-$n" "checkpoints/$ckpt" "$EVAL_GPU" "$base" \
      > "logs/eval_$tag-$n.log" 2>&1
  ) &
  EVAL_PID=$!
}

mkdir -p logs
[ -f "$LABELS" ] || { log "missing $LABELS"; exit 1; }
cleanup

# discard the invalid (reasoning-contaminated) sweep results
rm -rf results/sweep-3b-800 results/sweep-3b-2400 results/sweep-3b-4800 \
       results/sweep-7b-800 results/sweep-7b-2400

run_one "$B3" 800  sweep-3b q38c-3b-800
run_one "$B3" 2400 sweep-3b q38c-3b-2400
run_one "$B3" 4800 sweep-3b q38c-3b-4800
run_one "$B7" 800  sweep-7b q38c-7b-800
run_one "$B7" 2400 sweep-7b q38c-7b-2400
stop_eval
cleanup

log "sweep re-run complete"
.venv/bin/python3 make_overnight_reports.py >> "$LOG" 2>&1 || true
echo "done $(date -u '+%Y-%m-%d %H:%M UTC')" > logs/rerun_sweep.done
