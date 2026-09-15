#!/usr/bin/env bash
# Insurance for run_overnight2.sh: restart it if it dies before finishing.
# Resumable stages make a restart safe; flock in the driver prevents overlap.
cd /home/nick/ocr-bench || exit 1
for _ in $(seq 1 400); do
  if [ -f logs/overnight2.done ]; then
    echo "[$(date)] overnight2 complete; watchdog exit" >> logs/watchdog2.log
    exit 0
  fi
  if ! pgrep -f "[r]un_overnight2.sh" >/dev/null; then
    echo "[$(date)] supervisor not running - restarting" >> logs/watchdog2.log
    nohup setsid bash run_overnight2.sh >> logs/overnight2.log 2>&1 < /dev/null &
    sleep 20
  fi
  sleep 300
done
