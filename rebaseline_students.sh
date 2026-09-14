#!/usr/bin/env bash
# Re-evaluate the student models on a level playing field.
#
# The students were evaluated with eval_hf.py, which caps images at 1024*28*28
# (802,816 px). Every non-student model went through vLLM, which used the model's
# own default -- roughly 16x more pixels. So the students were not losing to
# dots-mocr and olmOCR-2 purely on merit; they were reading smaller images.
#
# Validation runs on the GT-labels adapter measured the size of that handicap:
# capped 0.2783 -> full resolution 0.2474, and that was while vLLM was also
# dropping the vision-tower LoRA. This re-runs every student at full resolution
# with the tower LoRA applied (LORA_TOWER=1), which is both the strongest
# configuration and the one comparable to the served models.
set -euo pipefail
cd "$(dirname "$0")"

GPU=${GPU:-0}
export LORA_TOWER=1          # apply vision-tower LoRA, as HF does
unset MAX_PIXELS 2>/dev/null || true   # full resolution, as the served models got

run_adapter() {  # run_adapter <result-name> <adapter-dir>
  local name=$1 adapter=$2
  if [ ! -d "$adapter" ]; then
    echo "### SKIP $name -- no adapter at $adapter"; return 0
  fi
  if [ -f "results/$name/summary.json" ]; then
    echo "### SKIP $name -- already evaluated"; return 0
  fi
  echo "### $name  <- $adapter"
  bash run_lora_model.sh "$name" "$adapter" "$GPU"
}

# Base model: no adapter, so it goes through the plain vLLM runner.
if [ ! -f results/qwen25vl3b-base-hires/summary.json ]; then
  echo "### qwen25vl3b-base-hires  <- no adapter (base model)"
  bash run_model.sh qwen25vl3b-base-hires Qwen/Qwen2.5-VL-3B-Instruct "$GPU" \
    "Extract the text content from this image."
fi

run_adapter qwen25vl3b-lora-hires          checkpoints/qwen25vl3b-lora
run_adapter qwen25vl3b-lora-teacher-hires  checkpoints/qwen25vl3b-lora-teacher
run_adapter qwen25vl3b-lora-q38-hires      checkpoints/qwen25vl3b-lora-q38

echo "=== re-baseline complete ==="
.venv/bin/python3 - <<'PY'
import json, os
rows = [
    ("base (no fine-tune)",        "qwen25vl3b-base",         "qwen25vl3b-base-hires"),
    ("LoRA, exact GT labels",      "qwen25vl3b-lora",         "qwen25vl3b-lora-hires"),
    ("LoRA, olmOCR-2 labels",      "qwen25vl3b-lora-teacher", "qwen25vl3b-lora-teacher-hires"),
    ("LoRA, Qwen3.8 labels",       None,                      "qwen25vl3b-lora-q38-hires"),
]
def cer(name):
    p = f"results/{name}/summary.json" if name else None
    if not p or not os.path.exists(p):
        return None
    return json.load(open(p))["mean_cer"]

print(f"\n{'student':<28}{'capped (old)':>14}{'full res':>12}{'gain':>10}")
print("-" * 64)
for label, old, new in rows:
    a, b = cer(old), cer(new)
    a_s = f"{a:.4f}" if a is not None else "--"
    b_s = f"{b:.4f}" if b is not None else "--"
    d_s = f"{a - b:+.4f}" if (a is not None and b is not None) else "--"
    print(f"{label:<28}{a_s:>14}{b_s:>12}{d_s:>10}")
PY
