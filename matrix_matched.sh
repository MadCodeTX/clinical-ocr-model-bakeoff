#!/usr/bin/env bash
# The distillation matrix, every run at the resolution the students were
# trained at.
#
# Why this exists: rebaseline_students.sh evaluated the adapters at vLLM's
# default resolution (12,845,056 px) while train_student.py trains at
# 1024*28*28 = 802,816 px -- a 16x train/eval mismatch. Under that mismatch the
# exact-GT adapter scored 0.2720 against a 0.2461 base, which reads as
# "distillation hurts" but actually measures an adapter applied outside its
# training distribution.
#
# Everything here runs at MAX_PIXELS=802816 with the vision-tower LoRA applied,
# so base and students are directly comparable and the only thing varying is
# the label source. Comparisons against the *served* models (dots-mocr,
# olmOCR-2, Qwen3.8) remain resolution-confounded and are not made here.
set -euo pipefail
cd "$(dirname "$0")"

GPU=${GPU:-0}
export LORA_TOWER=1
export MAX_PIXELS=802816

base_run() {  # base_run <result-name> <hf-id>
  [ -f "results/$1/summary.json" ] && { echo "### SKIP $1"; return 0; }
  echo "### $1  <- base $2 @ ${MAX_PIXELS}px"
  bash run_model.sh "$1" "$2" "$GPU" "Extract the text content from this image." \
    --mm-processor-kwargs "{\"max_pixels\":${MAX_PIXELS}}"
}

lora_run() {  # lora_run <result-name> <adapter> [base-hf-id]
  [ -f "results/$1/summary.json" ] && { echo "### SKIP $1"; return 0; }
  [ -d "$2" ] || { echo "### SKIP $1 -- no adapter $2"; return 0; }
  echo "### $1  <- $2 @ ${MAX_PIXELS}px"
  bash run_lora_model.sh "$1" "$2" "$GPU" "${3:-Qwen/Qwen2.5-VL-3B-Instruct}"
}

# ---- 3B family: one base, three label sources -------------------------------
base_run qwen25vl3b-base-m         Qwen/Qwen2.5-VL-3B-Instruct
lora_run qwen25vl3b-lora-gt-m      checkpoints/qwen25vl3b-lora
lora_run qwen25vl3b-lora-olmocr-m  checkpoints/qwen25vl3b-lora-teacher
lora_run qwen25vl3b-lora-q38-m     checkpoints/qwen25vl3b-lora-q38

# ---- 7B family: does a better teacher still help a stronger student? --------
base_run qwen25vl7b-base-m         Qwen/Qwen2.5-VL-7B-Instruct
lora_run qwen25vl7b-lora-q38-m     checkpoints/qwen25vl7b-lora-q38 Qwen/Qwen2.5-VL-7B-Instruct

echo "=== matched matrix complete ==="
.venv/bin/python3 - <<'PY'
import json, os

def s(name):
    p = f"results/{name}/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None

groups = [
    ("Qwen2.5-VL-3B", [
        ("base (no fine-tune)",  "qwen25vl3b-base-m"),
        ("LoRA, exact GT",       "qwen25vl3b-lora-gt-m"),
        ("LoRA, olmOCR-2 labels","qwen25vl3b-lora-olmocr-m"),
        ("LoRA, Qwen3.8 labels", "qwen25vl3b-lora-q38-m"),
    ]),
    ("Qwen2.5-VL-7B", [
        ("base (no fine-tune)",  "qwen25vl7b-base-m"),
        ("LoRA, Qwen3.8 labels", "qwen25vl7b-lora-q38-m"),
    ]),
]
SUB = ["normal", "handwriting", "poor", "rotated", "tables", "mixed"]
for title, rows in groups:
    print(f"\n{title}  (all runs @ 802,816 px, vision-tower LoRA applied)")
    print(f"{'variant':<26}{'mean CER':>10}{'vs base':>10}   " +
          "".join(f"{k:>13}" for k in SUB))
    print("-" * (46 + 13 * len(SUB)))
    base = s(rows[0][1])
    for label, name in rows:
        d = s(name)
        if not d:
            print(f"{label:<26}{'--':>10}")
            continue
        delta = f"{base['mean_cer'] - d['mean_cer']:+.4f}" if base and name != rows[0][1] else "--"
        subs = "".join(f"{d['per_subset'].get(k, {}).get('mean_cer', float('nan')):>13.4f}"
                       for k in SUB)
        print(f"{label:<26}{d['mean_cer']:>10.4f}{delta:>10}   {subs}")
PY
