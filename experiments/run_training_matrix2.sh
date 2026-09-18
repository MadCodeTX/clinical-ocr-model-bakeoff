#!/usr/bin/env bash
# Wave 2: build on the Wave-1 result that INPUT RESOLUTION was the best lever
# (7B @1.6M px = 0.0421 median, best student so far), plus the capacity /
# adaptation questions the theory raised.
#
#   * hires across seeds + corpus sizes   -> is resolution a robust win?
#   * hires + higher rank                  -> resolution x adaptation
#   * 3 epochs at the 800-doc sweet spot   -> does more optimisation help?
#   * full fine-tune of the 3B             -> does a stronger lever lift the 3B?
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/train_matrix2.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

B3=Qwen/Qwen2.5-VL-3B-Instruct
B7=Qwen/Qwen2.5-VL-7B-Instruct
D800=data/exp/train_synth_800.jsonl
D2400=data/exp/train_synth_2400.jsonl
D4400=data/exp/train_synth_4400.jsonl
D4400F=data/exp/train_synth_4400_filtered.jsonl
HI=1605632   # 2x the old 802,816 px budget

train_and_eval() {
  local lane=$1 name=$2 base=$3 data=$4 tpx=$5 epx=$6; shift 6
  local ckpt="checkpoints/y-$name"
  if [ ! -d "$ckpt" ]; then
    log "L$lane TRAIN y-$name (data=$(basename "$data") px=$tpx)"
    CUDA_VISIBLE_DEVICES=$lane train-venv/bin/python3 train_student2.py \
      --model "$base" --data "$data" --out "$ckpt" --max-pixels "$tpx" "$@" \
      > "logs/train_y-$name.log" 2>&1 || { log "L$lane TRAIN FAIL y-$name"; return 1; }
  fi
  if [ ! -f "results/y-$name/summary.json" ]; then
    log "L$lane EVAL  y-$name (px=$epx)"
    if [ -f "$ckpt/adapter_config.json" ]; then
      MAX_PIXELS=$epx LORA_TOWER=1 bash run_lora_model.sh "y-$name" "$ckpt" "$lane" "$base" \
        > "logs/eval_y-$name.log" 2>&1 || log "L$lane EVAL FAIL y-$name"
    else
      MAX_PIXELS=$epx bash run_full_model.sh "y-$name" "$ckpt" "$lane" \
        > "logs/eval_y-$name.log" 2>&1 || log "L$lane EVAL FAIL y-$name"
    fi
  fi
}

lane0() {
  train_and_eval 0 hires800-s0 "$B7" "$D800" "$HI" "$HI" --rank 16 --seed 0
  train_and_eval 0 hires800-s1 "$B7" "$D800" "$HI" "$HI" --rank 16 --seed 1
  train_and_eval 0 hires800-s2 "$B7" "$D800" "$HI" "$HI" --rank 16 --seed 2
  train_and_eval 0 hires2400  "$B7" "$D2400" "$HI" "$HI" --rank 16 --seed 0
  train_and_eval 0 ep3-800   "$B7" "$D800" 802816 802816 --epochs 3 --lr 5e-5 --max-chars 2000
}

lane1() {
  train_and_eval 1 hires4400-rank64 "$B7" "$D4400" "$HI" "$HI" --rank 64 --alpha 128 --seed 0
  train_and_eval 1 hires4400-filter "$B7" "$D4400F" "$HI" "$HI" --rank 16 --seed 0 --drop-repetition
  train_and_eval 1 3b-fullft-800  "$B3" "$D800" 802816 802816 --full-ft --optim adafactor --lr 1e-4
  train_and_eval 1 3b-fullft-4400 "$B3" "$D4400" 802816 802816 --full-ft --optim adafactor --lr 5e-5
  train_and_eval 1 3b-hires800    "$B3" "$D800" "$HI" "$HI" --rank 16 --seed 0
}

log "wave 2 start"
lane0 > logs/lane0b.log 2>&1 &
P0=$!
lane1 > logs/lane1b.log 2>&1 &
P1=$!
wait $P0 $P1
log "wave 2 complete"
echo "done $(date -u '+%Y-%m-%d %H:%M UTC')" > logs/train_matrix2.done
