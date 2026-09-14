#!/usr/bin/env bash
# Serve a LoRA student through vLLM and evaluate it, instead of generating with
# HF transformers (eval_hf.py). Same 328 documents, same prompt, same scorer --
# roughly 6x faster because vLLM batches continuously.
#
# IMPORTANT: our adapters contain vision-tower LoRA weights (target_modules
# match q_proj/k_proj/... inside the visual encoder as well as the language
# model). vLLM applies LoRA to the language model only. Before trusting any
# number from this path, run validate_lora_path.sh, which re-evaluates an
# adapter whose HF-generated score we already know and compares the two.
#
# Usage: run_lora_model.sh NAME ADAPTER_DIR GPU [BASE_MODEL]
set -euo pipefail
cd "$(dirname "$0")"

NAME=$1; ADAPTER=$2; GPU=$3
BASE=${4:-Qwen/Qwen2.5-VL-3B-Instruct}
PROMPT=${PROMPT:-"Extract the text content from this image."}
PORT=$((8100 + GPU))
CTR="ocr-lora-${NAME}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
ADAPTER_ABS=$(readlink -f "$ADAPTER")
RANK=$(python3 -c "import json;print(json.load(open('$ADAPTER_ABS/adapter_config.json'))['r'])")

echo "=== [$NAME] serving $BASE + LoRA $ADAPTER_ABS (rank $RANK) on GPU$GPU ==="
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" \
  --gpus "\"device=$GPU\"" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$ADAPTER_ABS":/adapter:ro \
  -p "$PORT":8000 \
  --ipc=host \
  vllm/vllm-openai:v0.27.1 \
  --model "$BASE" --served-model-name base \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --lora-modules "student=/adapter" \
  --max-lora-rank "$RANK" \
  --max-loras 1 > /dev/null

echo "=== [$NAME] waiting for model load ==="
ready=0
for i in $(seq 1 120); do
  if curl -s --max-time 2 "http://localhost:$PORT/v1/models" | grep -q student; then
    echo "=== [$NAME] ready after ~${i}0s ==="; ready=1; break
  fi
  # Surface a load failure instead of burning the full 20 minutes on it.
  if ! docker ps --format '{{.Names}}' | grep -q "^${CTR}$"; then
    echo "=== [$NAME] container exited during load ==="
    docker logs --tail 40 "$CTR"; exit 1
  fi
  sleep 10
done
if [ "$ready" != 1 ]; then
  echo "TIMEOUT waiting for $NAME"; docker logs --tail 40 "$CTR"; docker rm -f "$CTR" >/dev/null; exit 1
fi

# Record whether vLLM complained about adapter weights it could not apply.
docker logs "$CTR" 2>&1 | grep -iE "lora|adapter" | tail -20 > "logs/${NAME}-lora-load.log" || true

.venv/bin/python3 eval_cli.py --endpoint "http://localhost:$PORT" --model student \
  --name "$NAME" --prompt "$PROMPT" --out "results/$NAME" \
  --concurrency "${CONCURRENCY:-8}" --max-tokens "${MAX_TOKENS:-4096}"

docker logs "$CTR" > "results/$NAME/vllm.log" 2>&1 || true
docker rm -f "$CTR" >/dev/null
echo "=== [$NAME] done ==="
