#!/usr/bin/env bash
# Launch one OCR model with vLLM on a single GPU, wait for readiness, run eval, tear down.
# Usage: run_model.sh NAME HF_ID GPU PROMPT [EXTRA_VLLM_ARGS...]
set -euo pipefail
cd "$(dirname "$0")"

NAME=$1; HF_ID=$2; GPU=$3; PROMPT=$4; shift 4
PORT=$((8000 + GPU))
CTR="ocr-${NAME}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

echo "=== [$NAME] launching vLLM on GPU$GPU ($HF_ID) ==="
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" \
  --gpus "\"device=$GPU\"" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -p "$PORT":8000 \
  --ipc=host \
  vllm/vllm-openai:v0.27.1 \
  --model "$HF_ID" --served-model-name ocr \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  "$@" > /dev/null

echo "=== [$NAME] waiting for model load ==="
for i in $(seq 1 120); do
  if curl -s --max-time 2 "http://localhost:$PORT/v1/models" | grep -q ocr; then
    echo "=== [$NAME] ready after ~${i}0s ==="; break
  fi
  sleep 10
  if [ "$i" = 120 ]; then echo "TIMEOUT waiting for $NAME"; docker logs --tail 40 "$CTR"; exit 1; fi
done

.venv/bin/python3 eval_cli.py --endpoint "http://localhost:$PORT" --model ocr \
  --name "$NAME" --prompt "$PROMPT" --out "results/$NAME" \
  --concurrency "${CONCURRENCY:-8}" --max-tokens "${MAX_TOKENS:-4096}"

docker logs "$CTR" > "results/$NAME/vllm.log" 2>&1 || true
docker rm -f "$CTR" >/dev/null
echo "=== [$NAME] done ==="
