#!/usr/bin/env bash
# Train a 7B student on old synthetic (800) + new real medical scans (filtered),
# then evaluate on BOTH the old ClinOCR-Bench (independent GT) and the new
# medreal set (DeepSeek GT). Runs while a base-model eval owns GPU0.
#
# The real scans include a few very long pages (>20k chars) that blow up a
# 24 GB card at batch size 1; drop docs outside [40, MAXCHARS] chars.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/medmix_train.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

MAXCHARS=${MAXCHARS:-3000}
B7=Qwen/Qwen2.5-VL-7B-Instruct
MIX=/tmp/mixed_old800_new651.jsonl
CKPT=medmix-7b-old800-new

log "building filtered mixed corpus (max $MAXCHARS chars)"
.venv/bin/python3 - "$MAXCHARS" "$MIX" <<'PY'
import json, sys
maxchars, out_path = int(sys.argv[1]), sys.argv[2]
kept = dropped = 0
with open(out_path, "w") as out:
    for path in ("data/synth_large/teacher_labels_q38.jsonl", "data/medreal/labels.jsonl"):
        rows = [json.loads(l) for l in open(path)]
        if "synth" in path:
            rows = rows[:800]
        for r in rows:
            t = (r.get("text") or "").strip()
            if len(t) < 40 or len(t) > maxchars:
                dropped += 1
                continue
            out.write(json.dumps({"id": r["id"], "image": r["image"], "text": t,
                                  "handwriting": r.get("handwriting", False)}) + "\n")
            kept += 1
print(f"kept {kept}, dropped {dropped}")
PY

if [ ! -d "checkpoints/$CKPT" ]; then
  log "training 7B on mixed corpus (GPU1)"
  CUDA_VISIBLE_DEVICES=1 train-venv/bin/python3 train_student.py \
    --model "$B7" --data "$MIX" --out "checkpoints/$CKPT" --epochs 1 \
    > "logs/train_medmix_7b.log" 2>&1 || { log "TRAIN FAILED"; tail -25 logs/train_medmix_7b.log | tr '\r' '\n' | tail -8 | tee -a "$LOG"; }
fi

if [ -d "checkpoints/$CKPT" ]; then
  log "eval student on OLD ClinOCR (GPU0)"
  [ -f results/medmix-7b-old/summary.json ] || \
    LORA_TOWER=1 MAX_PIXELS=802816 DATA=data/clinocr \
      bash run_lora_model.sh medmix-7b-old "checkpoints/$CKPT" 0 "$B7"
  log "eval student on NEW medreal (GPU0)"
  [ -f results/medmix-7b-new/summary.json ] || \
    LORA_TOWER=1 MAX_PIXELS=802816 DATA=data/medreal \
      bash run_lora_model.sh medmix-7b-new "checkpoints/$CKPT" 0 "$B7"
fi

docker ps --format '{{.Names}}' | grep '^ocr-' | xargs -r docker rm -f >/dev/null 2>&1 || true
log "medmix training/eval complete"
echo "done $(date -u '+%Y-%m-%d %H:%M UTC')" > logs/medmix_train.done
