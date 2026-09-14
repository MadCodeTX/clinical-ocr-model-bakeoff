#!/usr/bin/env bash
# Restart the overnight supervisor + watchdog cleanly.
# Lives in a script so pkill patterns never match the invoking command line.
cd /home/nick/ocr-bench || exit 1

pkill -f 'bash overnight.sh' 2>/dev/null
pkill -f 'watchdog.sh' 2>/dev/null
pkill -f 'eval_paddle_pipeline.py' 2>/dev/null
pkill -f 'pipeline_run.sh' 2>/dev/null
sleep 3

docker rm -f ocr-paddleocr-vl >/dev/null 2>&1
rm -rf results/paddleocr-vl-pipeline

nohup setsid bash overnight.sh >> logs/supervisor.log 2>&1 < /dev/null &
sleep 2
nohup setsid ./watchdog.sh > /dev/null 2>&1 < /dev/null &

sleep 15
echo "--- supervisor tail:"
tail -3 logs/supervisor.log
echo "--- running:"
pgrep -af 'overnight.sh' | head -2
pgrep -af 'watchdog.sh' | head -1
