#!/usr/bin/env bash
# Shared helpers for experiment scripts.
ROOT=/home/nick/ocr-bench
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs results reports docs/samples

cleanup_containers() {
  docker ps --format '{{.Names}}' | grep '^ocr-' | xargs -r docker rm -f >/dev/null 2>&1 || true
}

score() {  # score <name>
  .venv/bin/python3 score_preds.py --preds "results/$1/predictions.jsonl" \
    --gt data/clinocr/eval.jsonl --out "results/$1" --name "$1"
}

copy_samples() {
  python3 - <<'PY'
import json, os, shutil
root = "/home/nick/ocr-bench"
gt = [json.loads(l) for l in open(os.path.join(root, "data/clinocr/eval.jsonl"))]
os.makedirs(os.path.join(root, "docs/samples"), exist_ok=True)
for sub in ("normal", "handwriting", "rotated", "mixed"):
    row = next((r for r in gt if r["subset"] == sub), None)
    if row:
        shutil.copy(os.path.join(root, row["image"]),
                    os.path.join(root, f"docs/samples/{sub}.jpg"))
PY
}
