#!/usr/bin/env bash
# Eval jinaai/jina-ocr-v1 on ClinOCR-Bench, pausing the user's production
# vLLM deployment (both GPUs) for the duration and restoring it afterwards.
set -uo pipefail
cd ~/ocr-bench

PROD=vllm-Qwen3.8-27B-FP8

restore() {
  echo "[jina-run] restarting production deployment $PROD"
  docker start "$PROD" && echo "[jina-run] production deployment restored"
}
trap restore EXIT

echo "[jina-run] stopping production deployment to free GPUs"
docker stop "$PROD"

echo "[jina-run] launching jina-ocr-v1 eval (gpu 0, concurrency 8)"
# Base weights with the FastMTP draft head stripped (vLLM's native
# DeepseekOCRForCausalLM rejects mtp_module.* params; MTP is acceleration-only,
# greedy outputs are identical). Served from the HF-cache mount.
PROMPT="Transcribe the provided document image into a clean Markdown format, preserving the natural reading order." \
  bash serve_and_eval.sh jina-ocr-v1 /root/.cache/huggingface/models/jina-ocr-v1-base 0 data/clinocr zero 8
rc=$?
echo "[jina-run] serve_and_eval rc=$rc"
exit $rc
