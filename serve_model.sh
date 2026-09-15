#!/usr/bin/env bash
# Serve a model with vLLM on one GPU and wait until ready (no eval, no teardown).
set -euo pipefail
NAME=$1; HF_ID=$2; GPU=$3
# GPU may be a single ordinal ("1") or a list ("0,1") for tensor parallelism.
# Port derives from the FIRST ordinal; TP size is the number of ordinals.
FIRST_GPU=${GPU%%,*}
TP=$(awk -F, '{print NF}' <<<"$GPU")
PORT=$((8000 + FIRST_GPU))
CTR="ocr-${NAME}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" --gpus "\"device=$GPU\"" \
  -v "$HF_CACHE":/root/.cache/huggingface -p "$PORT":8000 --ipc=host \
  vllm/vllm-openai:v0.27.1 \
  --model "$HF_ID" --served-model-name ocr ${EXTRA_SERVED:-} --trust-remote-code \
  --max-model-len "${MAX_LEN:-8192}" --tensor-parallel-size "$TP" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --gpu-memory-utilization "${GPU_MEM:-0.90}" > /dev/null
echo "waiting for $NAME on GPU(s) $GPU (TP=$TP) :8000->$PORT"
for i in $(seq 1 120); do
  if curl -s --max-time 2 "http://localhost:$PORT/v1/models" | grep -q ocr; then
    echo "$NAME ready"; exit 0
  fi
  sleep 10
done
echo "TIMEOUT $NAME"; docker logs --tail 30 "$CTR"; exit 1
