#!/usr/bin/env bash
# Usage: round.sh "NAME|HF_ID|GPU|PROMPT|MAXTOK" [...]
cd ~/ocr-bench
for spec in "$@"; do
  IFS="|" read -r NAME HF_ID GPU PROMPT MAXTOK <<< "$spec"
  MAX_TOKENS="$MAXTOK" bash run_model.sh "$NAME" "$HF_ID" "$GPU" "$PROMPT" > "logs/$NAME.log" 2>&1 &
done
wait
echo "ROUND COMPLETE" > logs/round.done
