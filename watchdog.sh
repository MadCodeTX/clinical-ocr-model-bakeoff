#!/usr/bin/env bash
# Insurance: restart the supervisor if it ever dies before finishing.
cd /home/nick/ocr-bench
for _ in $(seq 1 200); do
  if [ -f logs/overnight.done ]; then echo "[$(date)] overnight complete; watchdog exit" >> logs/watchdog.log; exit 0; fi
  if ! pgrep -f "[o]vernight.sh" > /dev/null; then
    echo "[$(date)] supervisor not running - restarting" >> logs/watchdog.log
    nohup setsid bash overnight.sh >> logs/supervisor.log 2>&1 < /dev/null &
  fi
  sleep 300
done
