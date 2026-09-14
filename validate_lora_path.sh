#!/usr/bin/env bash
# Gate for the fast eval path.
#
# run_lora_model.sh serves LoRA students through vLLM instead of generating with
# HF transformers. That is only a speedup if the two produce the same answer.
# Our adapters carry vision-tower LoRA weights that vLLM applies to the language
# model only, so "same answer" is an open question, not an assumption.
#
# This re-evaluates qwen25vl3b-lora -- whose HF-generated score is already on
# record -- through vLLM, and compares. If the two agree, every later run can
# use the fast path and still be compared against the existing baselines. If
# they disagree, the matrix has to stay on eval_hf.py and the disagreement is
# itself a finding worth writing down.
set -euo pipefail
cd "$(dirname "$0")"

GPU=${1:-0}
REF=results/qwen25vl3b-lora/summary.json
TOL=${TOL:-0.010}

[ -f "$REF" ] || { echo "missing reference $REF"; exit 1; }

bash run_lora_model.sh qwen25vl3b-lora-vllm checkpoints/qwen25vl3b-lora "$GPU"

.venv/bin/python3 - "$REF" "$TOL" <<'PY'
import json, sys

ref = json.load(open(sys.argv[1]))
new = json.load(open("results/qwen25vl3b-lora-vllm/summary.json"))
tol = float(sys.argv[2])

d = new["mean_cer"] - ref["mean_cer"]
print(f"\n  HF transformers (reference) : {ref['mean_cer']:.4f} mean CER, "
      f"{ref['pages_per_sec']:.3f} pages/s")
print(f"  vLLM + LoRA (fast path)     : {new['mean_cer']:.4f} mean CER, "
      f"{new['pages_per_sec']:.3f} pages/s")
print(f"  delta                       : {d:+.4f}  (tolerance ±{tol:.3f})")
speedup = new["pages_per_sec"] / ref["pages_per_sec"] if ref["pages_per_sec"] else float("nan")
print(f"  speedup                     : {speedup:.1f}x")

print("\n  per subset:")
for k in sorted(set(ref["per_subset"]) | set(new["per_subset"])):
    a = ref["per_subset"].get(k, {}).get("mean_cer")
    b = new["per_subset"].get(k, {}).get("mean_cer")
    if a is None or b is None:
        continue
    print(f"    {k:<14} {a:.4f} -> {b:.4f}  ({b - a:+.4f})")

if abs(d) <= tol:
    print("\nPASS: the two paths agree. Use vLLM serving for the matrix.")
else:
    print("\nFAIL: the paths disagree beyond tolerance.")
    print("Most likely vLLM is not applying the vision-tower LoRA weights that")
    print("HF applies. Keep the matrix on eval_hf.py, or retrain with")
    print("target_modules restricted to the language model so both agree.")
    sys.exit(1)
PY
