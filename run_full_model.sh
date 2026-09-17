#!/usr/bin/env bash
# Serve a FULLY fine-tuned checkpoint (local dir) with vLLM and evaluate it.
# Mirrors run_lora_model.sh but mounts the model dir and serves it directly.
#
# Usage: run_full_model.sh NAME MODEL_DIR GPU [DATA]
set -uo pipefail
cd "$(dirname "$0")"

NAME=$1; MODEL_DIR=$2; GPU=$3; DATA=${DATA:-data/clinocr}
PROMPT=${PROMPT:-"Extract the text content from this image."}
PORT=$((8100 + GPU))
CTR="ocr-full-${NAME}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODEL_ABS=$(readlink -f "$MODEL_DIR")
PIXELS_ARG=""
[ -n "${MAX_PIXELS:-}" ] && PIXELS_ARG="--mm-processor-kwargs {\"max_pixels\":${MAX_PIXELS}}"

[ -f "results/$NAME/summary.json" ] && { echo "### SKIP $NAME (done)"; exit 0; }
echo "=== [$NAME] serving full FT $MODEL_ABS on GPU$GPU ==="
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" \
  --gpus "\"device=$GPU\"" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$MODEL_ABS":/model:ro \
  -p "$PORT":8000 --ipc=host \
  vllm/vllm-openai:v0.27.1 \
  --model /model --served-model-name student --trust-remote-code \
  --max-model-len 8192 --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --gpu-memory-utilization 0.90 ${PIXELS_ARG} > /dev/null

ready=0
for i in $(seq 1 120); do
  if curl -s --max-time 2 "http://localhost:$PORT/v1/models" | grep -q student; then
    echo "=== [$NAME] ready after ~${i}0s ==="; ready=1; break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CTR}$"; then
    echo "=== [$NAME] container exited during load ==="; docker logs --tail 30 "$CTR"; exit 1
  fi
  sleep 10
done
[ "$ready" = 1 ] || { echo "TIMEOUT $NAME"; docker logs --tail 30 "$CTR"; docker rm -f "$CTR"; exit 1; }

.venv/bin/python3 eval_cli.py --endpoint "http://localhost:$PORT" --model student \
  --name "$NAME" --prompt "$PROMPT" --data "$DATA" --out "results/$NAME" \
  --concurrency "${CONCURRENCY:-8}" --max-tokens "${MAX_TOKENS:-4096}" \
  ${NO_REPEAT_NGRAM:+--no-repeat-ngram-size "$NO_REPEAT_NGRAM"} --resume

mkdir -p "results/$NAME"
docker logs "$CTR" > "results/$NAME/vllm.log" 2>&1 || true
docker rm -f "$CTR" >/dev/null 2>&1 || true
echo "### $NAME done"
