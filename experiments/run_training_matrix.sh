#!/usr/bin/env bash
# Wave 1 of "can we make training work better?" experiments.
#
# Each experiment trains a student (if its checkpoint is absent) and evaluates it
# on ClinOCR-Bench (if its result is absent), all within one GPU lane so nothing
# contends. Results land in results/x-<name>/summary.json.
#
# Levers under test, mapped to the failure analysis:
#   * seeds / corpus size   -> is the 7B gain real and does more data help?
#   * resolution            -> is it a perception (information) limit?
#   * LoRA rank / scope     -> is the low-rank update the bottleneck?
#   * full fine-tune (3B)   -> does a stronger adaptation lift the 3B?
#   * label filtering       -> does teacher noise cap the student?
#   * real+synth mixture    -> does real data transfer?
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/train_matrix.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

B3=Qwen/Qwen2.5-VL-3B-Instruct
B7=Qwen/Qwen2.5-VL-7B-Instruct
D800=data/exp/train_synth_800.jsonl
D2400=data/exp/train_synth_2400.jsonl
D4400=data/exp/train_synth_4400.jsonl
D4400F=data/exp/train_synth_4400_filtered.jsonl
REAL=data/medreal/labels.jsonl
VAL=data/exp/val_synth.jsonl

# train_and_eval LANE NAME BASE DATA TRAIN_PX EVAL_PX [extra train args...]
train_and_eval() {
  local lane=$1 name=$2 base=$3 data=$4 tpx=$5 epx=$6; shift 6
  local ckpt="checkpoints/x-$name"
  if [ ! -d "$ckpt" ]; then
    log "L$lane TRAIN x-$name (base=$base data=$(basename "$data") px=$tpx)"
    CUDA_VISIBLE_DEVICES=$lane train-venv/bin/python3 train_student2.py \
      --model "$base" --data "$data" --out "$ckpt" --max-pixels "$tpx" "$@" \
      > "logs/train_x-$name.log" 2>&1 || { log "L$lane TRAIN FAIL x-$name"; return 1; }
  fi
  if [ ! -f "results/x-$name/summary.json" ]; then
    log "L$lane EVAL  x-$name (px=$epx)"
    if [ -f "$ckpt/adapter_config.json" ]; then
      MAX_PIXELS=$epx LORA_TOWER=1 bash run_lora_model.sh "x-$name" "$ckpt" "$lane" "$base" \
        > "logs/eval_x-$name.log" 2>&1 || log "L$lane EVAL FAIL x-$name"
    else
      MAX_PIXELS=$epx bash run_full_model.sh "x-$name" "$ckpt" "$lane" \
        > "logs/eval_x-$name.log" 2>&1 || log "L$lane EVAL FAIL x-$name"
    fi
  fi
}

lane0() {
  train_and_eval 0 7b800-s0  "$B7" "$D800"  802816 802816 --rank 16 --seed 0
  train_and_eval 0 7b800-s1  "$B7" "$D800"  802816 802816 --rank 16 --seed 1
  train_and_eval 0 7b800-s2  "$B7" "$D800"  802816 802816 --rank 16 --seed 2
  train_and_eval 0 7b2400-s0 "$B7" "$D2400" 802816 802816 --rank 16 --seed 0
  train_and_eval 0 7b2400-s1 "$B7" "$D2400" 802816 802816 --rank 16 --seed 1
  train_and_eval 0 7b-hires  "$B7" "$D4400" 1605632 1605632 --rank 16 --seed 0
  train_and_eval 0 7b-3ep    "$B7" "$D4400" 802816 802816 --rank 16 --epochs 3 --lr 5e-5 \
    --val-file "$VAL" --save-best
}

lane1() {
  train_and_eval 1 7b4400-s0 "$B7" "$D4400" 802816 802816 --rank 16 --seed 0
  train_and_eval 1 7b4400-s1 "$B7" "$D4400" 802816 802816 --rank 16 --seed 1
  train_and_eval 1 7b-rank64 "$B7" "$D4400" 802816 802816 --rank 64 --alpha 128 --seed 0
  train_and_eval 1 7b-vision "$B7" "$D4400" 802816 802816 --target-scope vision --rank 16 --seed 0
  train_and_eval 1 7b-filter "$B7" "$D4400F" 802816 802816 --rank 16 --seed 0 --drop-repetition
  train_and_eval 1 3b-fullft-800  "$B3" "$D800"  802816 802816 --full-ft --optim adafactor --lr 1e-4
  train_and_eval 1 3b-fullft-4400 "$B3" "$D4400" 802816 802816 --full-ft --optim adafactor --lr 5e-5
}

log "wave 1 start"
lane0 > logs/lane0.log 2>&1 &
P0=$!
lane1 > logs/lane1.log 2>&1 &
P1=$!
wait $P0 $P1
log "wave 1 complete"
.venv/bin/python3 make_overnight_reports.py >> "$LOG" 2>&1 || true
echo "done $(date -u '+%Y-%m-%d %H:%M UTC')" > logs/train_matrix.done
