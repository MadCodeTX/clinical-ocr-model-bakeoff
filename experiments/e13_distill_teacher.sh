#!/usr/bin/env bash
# Distillation from teacher labels: mint olmOCR-2 labels for the whole synthetic
# corpus (unlabeled-scan simulation), train a second student on them, evaluate.
source "$(dirname "$0")/_common.sh"
set -o pipefail
cleanup_containers

# 1) teacher labels for all 800 synthetic docs
bash serve_model.sh olmocr-2 allenai/olmOCR-2-7B-1025 1
.venv/bin/python3 label_synth.py --endpoint http://localhost:8001 \
  --prompt "Extract the contents. [Markdown]." --data data/synth/labels.jsonl \
  --n 800 --out results/teacher-labels-full
cleanup_containers

# 2) build teacher-label training set
.venv/bin/python3 - <<'PY'
import json
root = "/home/nick/ocr-bench"
preds = {json.loads(l)["doc_id"]: json.loads(l)
         for l in open(f"{root}/results/teacher-labels-full/predictions.jsonl")}
rows = [json.loads(l) for l in open(f"{root}/data/synth/labels.jsonl")]
n = 0
with open(f"{root}/data/synth/teacher_labels.jsonl", "w") as f:
    for r in rows:
        p = preds.get(r["id"])
        if p and p.get("prediction"):
            f.write(json.dumps({"id": r["id"], "image": r["image"],
                                "text": p["prediction"],
                                "handwriting": r["handwriting"]}) + "\n")
            n += 1
print("teacher-label training docs:", n)
PY

# 3) train + evaluate
CUDA_VISIBLE_DEVICES=1 train-venv/bin/python3 train_student.py \
  --data data/synth/teacher_labels.jsonl \
  --out checkpoints/qwen25vl3b-lora-teacher --epochs 1
CUDA_VISIBLE_DEVICES=0 train-venv/bin/python3 eval_hf.py \
  --out results/qwen25vl3b-lora-teacher --adapter checkpoints/qwen25vl3b-lora-teacher
score qwen25vl3b-lora-teacher
