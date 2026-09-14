#!/usr/bin/env bash
# Overnight supervisor. Runs the experiment queue sequentially, each time-boxed,
# then regenerates reports/REPORT.md and pushes to GitHub. Survives SSH logout.
# Resumable: experiments with results/<name>/.done are skipped.
cd /home/nick/ocr-bench || exit 1
mkdir -p logs results reports docs/samples

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
NOW=$(date +%s)
DEADLINE=$(date -d "today 07:00" +%s)
[ "$NOW" -gt "$DEADLINE" ] && DEADLINE=$(date -d "tomorrow 07:00" +%s)
echo "[$(date)] supervisor start; deadline $(date -d @$DEADLINE)"

# report deps (idempotent)
.venv/bin/python3 -c "import matplotlib" 2>/dev/null || \
  .venv/bin/pip install -q matplotlib

wait_gpus() {
  local ok=0
  for _ in $(seq 1 120); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
    if [ "$(echo "$used" | awk '{print ($1<2000 && $2<2000)}')" = "1" ]; then
      ok=$((ok+1)); [ $ok -ge 2 ] && return 0
    else
      ok=0
    fi
    sleep 15
  done
  echo "[$(date)] warn: GPUs still busy after wait"
  return 0
}

cleanup_containers() {
  docker ps --format '{{.Names}}' | grep '^ocr-' | xargs -r docker rm -f >/dev/null 2>&1 || true
}

publish() {
  .venv/bin/python3 make_report.py > logs/report.log 2>&1 || echo "report failed"
  git add -A >/dev/null 2>&1
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -qm "results: $(date -u '+%Y-%m-%d %H:%M UTC') update" || true
  fi
  git push -q origin main 2>>logs/git.log || echo "[$(date)] push failed"
}

while read -r name budget; do
  [ -z "$name" ] && continue
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "[$(date)] deadline reached"; break; }
  if [ -f "results/$name/.done" ]; then
    echo "[$(date)] skip $name (done)"; continue
  fi
  cleanup_containers
  wait_gpus
  echo "[$(date)] RUN $name (budget ${budget}s)"
  start=$(date +%s)
  timeout "$budget" bash "experiments/$name.sh" > "logs/$name.log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - start ))
  mkdir -p "results/$name"
  echo "$rc $secs" > "results/$name/.rc"
  [ $rc -eq 0 ] && touch "results/$name/.done"
  echo "[$(date)] END $name rc=$rc ${secs}s"
  cleanup_containers
  publish
done < experiments/queue.txt

publish
echo "[$(date)] ALL DONE" > logs/overnight.done
