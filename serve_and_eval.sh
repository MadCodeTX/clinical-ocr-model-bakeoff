#!/usr/bin/env bash
# Serve a base (non-LoRA) VLM with vLLM and run eval_cli against it for a chosen
# dataset / one-shot regime, then tear down. Used for the one-shot study (A2)
# and the OmniDocBench generalisation study (A3), which need --shot / --data that
# run_model.sh does not pass.
#
# Usage: serve_and_eval.sh NAME HF_ID GPU DATA SHOT [CONCURRENCY]
#   GPU may be "0" or "0,1" (tensor parallel).
set -uo pipefail
cd "$(dirname "$0")"

NAME=$1; HF_ID=$2; GPU=$3; DATA=${4:-data/clinocr}; SHOT=${5:-zero}; CONC=${6:-8}
FIRST=${GPU%%,*}; PORT=$((8000 + FIRST))
PROMPT=${PROMPT:-"Extract the text content from this image."}
CTR="ocr-$NAME"

[ -f "results/$NAME/summary.json" ] && { echo "### SKIP $NAME (done)"; exit 0; }

bash serve_model.sh "$NAME" "$HF_ID" "$GPU" || { echo "serve failed"; exit 1; }

.venv/bin/python3 eval_cli.py --endpoint "http://localhost:$PORT" --model ocr \
  --name "$NAME" --prompt "$PROMPT" --data "$DATA" --shot "$SHOT" \
  --out "results/$NAME" --concurrency "$CONC" \
  --max-tokens "${MAX_TOKENS:-4096}" --resume

mkdir -p "results/$NAME"
docker logs "$CTR" > "results/$NAME/vllm.log" 2>&1 || true
docker rm -f "$CTR" >/dev/null 2>&1 || true
echo "### $NAME done"
