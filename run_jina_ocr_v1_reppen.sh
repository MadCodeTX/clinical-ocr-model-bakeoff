#!/usr/bin/env bash
# Eval jina-ocr-v1 with its recommended serving settings (repetition_penalty=1.05)
# per the model card's vllm_sampling_params(). Pauses/restores the prod deployment.
set -uo pipefail
cd ~/ocr-bench

NAME=jina-ocr-v1-reppen
MODEL=/root/.cache/huggingface/models/jina-ocr-v1-base
PROD=vllm-Qwen3.8-27B-FP8
PROMPT="Transcribe the provided document image into a clean Markdown format, preserving the natural reading order."

restore() {
  echo "[$NAME] restarting production deployment"
  docker start "$PROD" && echo "[$NAME] production deployment restored"
}
trap restore EXIT

[ -f "results/$NAME/summary.json" ] && { echo "done already"; exit 0; }

echo "[$NAME] stopping production deployment"
docker stop "$PROD"

bash serve_model.sh "$NAME" "$MODEL" 0 || { echo "serve failed"; exit 1; }

.venv/bin/python3 eval_cli.py --endpoint "http://localhost:8000" --model ocr \
  --name "$NAME" --prompt "$PROMPT" --data data/clinocr --shot zero \
  --out "results/$NAME" --concurrency 8 --max-tokens 4096 \
  --repetition-penalty 1.05 --resume

mkdir -p "results/$NAME"
docker logs "ocr-$NAME" > "results/$NAME/vllm.log" 2>&1 || true
docker rm -f "ocr-$NAME" >/dev/null 2>&1 || true
echo "[$NAME] done"
