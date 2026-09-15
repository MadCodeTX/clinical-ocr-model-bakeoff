#!/usr/bin/env bash
# Does adding real medical-scan data help the distilled student?
#
# New corpus: data/medreal (651 pages from public HF medical datasets, DeepSeek
# labels). We (a) evaluate base models on the new set, (b) train a 7B student on
# old synthetic (800) + new real (651) and (c) evaluate that student on BOTH the
# old ClinOCR-Bench (independent GT) and the new medreal set (DeepSeek GT).
#
# Note on validity: the new-set ground truth is the DeepSeek label we trained on,
# so a distilled student scores well there partly by construction. The old set
# (human-audited GT) is the honest test of whether the extra data helped.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/medreal_experiment.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

B7=Qwen/Qwen2.5-VL-7B-Instruct
PIX='--mm-processor-kwargs {"max_pixels":802816}'
OLDLABELS=data/synth_large/teacher_labels_q38.jsonl
NEWLABELS=data/medreal/labels.jsonl
MIX=/tmp/mixed_old800_new651.jsonl
CKPT=medmix-7b-old800-new

cleanup() { docker ps --format '{{.Names}}' | grep '^ocr-' | xargs -r docker rm -f >/dev/null 2>&1 || true; }
cleanup

# mixed training corpus: 800 old synthetic + all new real
head -800 "$OLDLABELS" > "$MIX"
cat "$NEWLABELS" >> "$MIX"
log "mixed corpus: $(wc -l < "$MIX") docs"

# 1) base 7B on the NEW set (matched resolution) -- GPU0, while we train on GPU1
if [ ! -f results/medreal-qwen25vl7b-base/summary.json ]; then
  log "eval base 7B on medreal (GPU0)"
  ( EXTRA_SERVED="$PIX" bash serve_and_eval.sh medreal-qwen25vl7b-base "$B7" 0 data/medreal zero 8 \
      > logs/eval_medreal-base.log 2>&1 ) &
  BASE_PID=$!
else
  BASE_PID=""
fi

# 2) train the mixed student on GPU1 (overlaps the GPU0 base eval above)
if [ ! -d "checkpoints/$CKPT" ]; then
  log "train 7B on mixed corpus -> checkpoints/$CKPT"
  CUDA_VISIBLE_DEVICES=1 train-venv/bin/python3 train_student.py \
    --model "$B7" --data "$MIX" --out "checkpoints/$CKPT" --epochs 1 \
    > "logs/train_medmix_7b.log" 2>&1 || log "TRAIN FAILED (see logs/train_medmix_7b.log)"
fi
[ -n "${BASE_PID:-}" ] && wait "$BASE_PID" 2>/dev/null || true
cleanup

# 3) student on OLD (independent GT) then NEW (DeepSeek GT), GPU0
if [ -d "checkpoints/$CKPT" ]; then
  [ -f results/medmix-7b-old/summary.json ] || \
    LORA_TOWER=1 MAX_PIXELS=802816 DATA=data/clinocr \
      bash run_lora_model.sh medmix-7b-old "checkpoints/$CKPT" 0 "$B7"
  [ -f results/medmix-7b-new/summary.json ] || \
    LORA_TOWER=1 MAX_PIXELS=802816 DATA=data/medreal \
      bash run_lora_model.sh medmix-7b-new "checkpoints/$CKPT" 0 "$B7"
fi

# 4) a non-distilled structure model on the NEW set, for reference -- GPU1
if [ ! -f results/medreal-dots-mocr/summary.json ]; then
  log "eval dots.mocr on medreal (GPU1)"
  bash serve_and_eval.sh medreal-dots-mocr rednote-hilab/dots.mocr 1 data/medreal zero 8 \
    > logs/eval_medreal-dots.log 2>&1 || true
fi

cleanup
log "medreal experiment complete"
echo "done $(date -u '+%Y-%m-%d %H:%M UTC')" > logs/medreal_experiment.done
