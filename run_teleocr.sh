#!/usr/bin/env bash
# Eval StarDoc-AI/TeleOCR (1.4B, Qwen2.5-VL arch) on ClinOCR-Bench, pausing the
# user's production vLLM deployment for the duration and restoring it after.
set -uo pipefail
cd ~/ocr-bench

PROD=vllm-Qwen3.8-27B-FP8

restore() {
  echo "[teleocr] restarting production deployment $PROD"
  docker start "$PROD" && echo "[teleocr] production deployment restored"
}
trap restore EXIT

echo "[teleocr] stopping production deployment to free GPUs"
docker stop "$PROD"

echo "[teleocr] launching TeleOCR eval (gpu 0, concurrency 8)"
# TeleOCR config quirk: hidden_size=1024 / 16 heads but head_dim=128. vLLM's
# native Qwen2.5-VL impl derives head_dim = hidden//heads = 64 and trips the
# M-RoPE assertion; the Transformers impl honors config.head_dim, so serve via
# --model-impl transformers.
# TeleOCR quirks: (1) config has hidden_size=1024 / 16 heads but head_dim=128;
# vLLM's native Qwen2.5-VL impl derives head_dim = hidden//heads = 64 and trips
# the M-RoPE assertion. (2) The repo's custom modeling_naviocr.py crashes with
# transformers' current ROPE_INIT_FUNCTIONS (KeyError 'default'). Fix: local
# copy with auto_map stripped (stock Qwen2.5-VL arch) + --model-impl
# transformers, which honors config.head_dim.
# transformers 5.x compat: also disable torch.compile tracing of the custom code.
EXTRA_SERVED="--model-impl transformers --enforce-eager" \
PROMPT="Please output the text content from the image." \
  bash serve_and_eval.sh teleocr /root/.cache/huggingface/models/teleocr-stock 0 data/clinocr zero 8
rc=$?
echo "[teleocr] serve_and_eval rc=$rc"
exit $rc
