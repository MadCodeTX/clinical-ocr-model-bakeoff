#!/usr/bin/env bash
# Overnight OCR run #2 -- implements OVERNIGHT_PLAN.md, priority-ordered and
# resumable. Every stage is a no-op if its results already exist, so it is safe
# to re-run after a crash. Two GPUs are used as two lanes where possible; the
# 27B teacher minting (B1) briefly needs both and runs between stages.
#
#   A1  noise floor, n=5 (GPU0 dots.mocr || GPU1 student)
#   B1  generate + mint the 4800-doc teacher corpus (both GPUs, TP=2)
#   B2  corpus-size sweep, GPU1        (parallel)
#   A2  one-shot regimes,      GPU0    (parallel)
#   A3  OmniDocBench,          both     (only if time remains)
#   B3  runaway mitigation              (only if time remains)
#
# Usage: bash run_overnight2.sh          (nohup/tmux for unattended)
#
# Env: DEADLINE_EPOCH  unix ts to stop by (default: next 07:00)
#      SKIP_NOISE=1    do not (re)run A1
#      MINT_N=4800     teacher labels to mint
set -uo pipefail
cd "$(dirname "$0")" || exit 1

# Single instance: the watchdog may try to restart us while a run is live.
exec 8>logs/overnight2.lock
if ! flock -n 8; then
  echo "run_overnight2.sh already active; exiting"
  exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs results reports

NOW=$(date +%s)
DEADLINE=${DEADLINE_EPOCH:-$(date -d "today 07:00" +%s)}
[ "$NOW" -gt "$DEADLINE" ] && DEADLINE=$(date -d "tomorrow 07:00" +%s)
PROMPT="Extract the text content from this image."

log()  { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
has()  { [ -f "results/$1/summary.json" ]; }
has_ckpt() { [ -d "checkpoints/$1" ]; }

cleanup_containers() {
  docker ps --format '{{.Names}}' | grep '^ocr-' | xargs -r docker rm -f >/dev/null 2>&1 || true
}

wait_gpus_free() {
  local tries=${1:-180}   # 180 * 15s = 45 min
  for _ in $(seq 1 "$tries"); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
    if [ "$(echo "$used" | awk '{print ($1<2500 && $2<2500)}')" = "1" ]; then
      return 0
    fi
    sleep 15
  done
  log "WARN: GPUs still busy after wait; forcing cleanup"
  cleanup_containers
  sleep 20
}

# Wait for a run that another lane has already started (identified by its
# vLLM container) instead of launching a duplicate.
wait_or_run() {  # wait_or_run NAME  (remaining args: the command)
  local name=$1; shift
  has "$name" && return 0
  if docker ps --format '{{.Names}}' | grep -q -- "-${name}\$"; then
    log "$name already running; waiting"
    while ! has "$name" && docker ps --format '{{.Names}}' | grep -q -- "-${name}\$"; do
      sleep 15
    done
    return 0
  fi
  "$@"
}

publish() {
  .venv/bin/python3 make_overnight_reports.py > logs/overnight_reports.log 2>&1 \
    || log "overnight reports failed"
  .venv/bin/python3 make_report.py > logs/report.log 2>&1 || log "make_report failed"
  git add -A >/dev/null 2>&1
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -qm "overnight run: $(date -u '+%Y-%m-%d %H:%M UTC')" || true
  fi
  git push -q origin main 2>>logs/git.log || log "push failed"
}

# ---------------------------------------------------------------------------
# A1  noise floor
# ---------------------------------------------------------------------------
stage_noise() {
  [ "${SKIP_NOISE:-0}" = "1" ] && { log "A1 skipped"; return 0; }
  log "STAGE A1 noise floor"
  ( for i in 1 2 3 4 5; do
      wait_or_run "noise-dotsmocr-$i" env GPU=0 bash run_model.sh \
        "noise-dotsmocr-$i" rednote-hilab/dots.mocr 0 "$PROMPT"
    done ) > logs/a1_dots.log 2>&1 &
  local a=$!
  ( for i in 1 2 3 4 5; do
      wait_or_run "noise-student-$i" env LORA_TOWER=1 MAX_PIXELS=802816 \
        bash run_lora_model.sh "noise-student-$i" checkpoints/qwen25vl3b-lora 1
    done ) > logs/a1_student.log 2>&1 &
  local b=$!
  wait "$a" "$b"
  cleanup_containers
  publish
}

# ---------------------------------------------------------------------------
# B1  generate + mint the teacher corpus
# ---------------------------------------------------------------------------
ensure_synth() {
  local target=${1:-4800}
  local f=data/synth_large/labels.jsonl
  local n
  n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
  if [ "$n" -ge "$target" ]; then return 0; fi
  if pgrep -f "gen_synth.py --n ${target}" >/dev/null; then
    log "waiting for running gen_synth ($n/$target)"
    while pgrep -f "gen_synth.py --n ${target}" >/dev/null; do sleep 30; done
    n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
  fi
  if [ "$n" -lt "$target" ]; then
    log "generating synthetic corpus ($target docs)"
    .venv/bin/python3 gen_synth.py --n "$target" --out data/synth_large --seed 11 \
      > logs/gen_synth_large.log 2>&1 || { log "gen_synth failed"; return 1; }
  fi
}

stage_mint() {
  local MINT_N=${MINT_N:-4800}
  ensure_synth "$MINT_N" || return 1
  local have
  have=$( [ -f data/synth_large/teacher_labels_q38.jsonl ] && \
          wc -l < data/synth_large/teacher_labels_q38.jsonl || echo 0 )
  [ "$have" -ge 800 ] && { log "B1 already minted ($have labels)"; return 0; }

  # Cap the mint so the sweep and one-shot studies still fit.
  local tl; tl=$(left)
  local budget=$(( tl - 4*3600 ))
  [ "$budget" -lt 1200 ] && budget=1200
  [ "$budget" -gt 3*3600 ] && budget=$((3*3600))
  local want=$MINT_N
  [ "$budget" -lt 5400 ] && want=2400   # <1.5h of mint job -> fewer docs

  log "STAGE B1 mint $want teacher labels (budget ${budget}s, ${tl}s left)"
  # An A1 lane may still be running (e.g. SKIP_NOISE=1 while the noise runs
  # were started separately); let it finish rather than reaping its containers.
  while pgrep -f "run_model.sh noise-|run_lora_model.sh noise-" >/dev/null; do
    log "  A1 lane still running; waiting before mint"
    sleep 30
  done
  wait_gpus_free
  bash serve_model.sh qwen38-27b Qwen/Qwen3.8-27B-FP8 0,1 || return 1
  timeout "$budget" .venv/bin/python3 label_synth.py --resume \
    --endpoint http://localhost:8000 --model ocr --teacher-name qwen38-27b \
    --prompt "$PROMPT" --data data/synth_large/labels.jsonl --n "$want" \
    --concurrency "${MINT_CONC:-8}" --out results/teacher-labels-q38-4800 \
    >> logs/mint_q38.log 2>&1
  log "mint rc=$? (timeout 124 means partial; resumable)"
  docker rm -f ocr-qwen38-27b >/dev/null 2>&1 || true
  cleanup_containers

  .venv/bin/python3 build_teacher_set.py \
    --preds results/teacher-labels-q38-4800/predictions.jsonl \
    --labels data/synth_large/labels.jsonl \
    --out data/synth_large/teacher_labels_q38.jsonl \
    >> logs/mint_q38.log 2>&1 || log "build_teacher_set failed"
  publish
}

# ---------------------------------------------------------------------------
# B2  corpus-size sweep (GPU1)
# ---------------------------------------------------------------------------
stage_sweep() {
  log "STAGE B2 corpus sweep"
  local labels=data/synth_large/teacher_labels_q38.jsonl
  [ -f "$labels" ] || { log "no teacher labels; skipping sweep"; return 1; }
  local avail; avail=$(wc -l < "$labels")
  local tl; tl=$(left)
  for n in 800 2400 4800; do
    [ "$n" -gt "$avail" ] && continue
    # if fewer than ~2.5h remain, stop adding larger points
    [ "$tl" -lt 9000 ] && [ "$n" -gt 800 ] && continue
    if ! has_ckpt "q38-3b-$n"; then
      head -"$n" "$labels" > "/tmp/tl_$n.jsonl"
      log "  training 3B on $n docs"
      CUDA_VISIBLE_DEVICES=1 train-venv/bin/python3 train_student.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct --data "/tmp/tl_$n.jsonl" \
        --out "checkpoints/q38-3b-$n" --epochs 1 \
        > "logs/train_q38_3b_$n.log" 2>&1 || { log "train $n failed"; continue; }
    fi
    has "sweep-3b-$n" || \
      LORA_TOWER=1 MAX_PIXELS=802816 bash run_lora_model.sh \
        "sweep-3b-$n" "checkpoints/q38-3b-$n" 1
  done

  # 7B only if a lot of time is left (it trains ~2x slower)
  if [ "$(left)" -gt 9000 ] && [ "$avail" -ge 2400 ]; then
    for n in 800 2400; do
      [ "$n" -gt "$avail" ] && continue
      if ! has_ckpt "q38-7b-$n"; then
        head -"$n" "$labels" > "/tmp/tl7b_$n.jsonl"
        log "  training 7B on $n docs"
        CUDA_VISIBLE_DEVICES=1 train-venv/bin/python3 train_student.py \
          --model Qwen/Qwen2.5-VL-7B-Instruct --data "/tmp/tl7b_$n.jsonl" \
          --out "checkpoints/q38-7b-$n" --epochs 1 \
          > "logs/train_q38_7b_$n.log" 2>&1 || continue
      fi
      has "sweep-7b-$n" || \
        LORA_TOWER=1 MAX_PIXELS=802816 bash run_lora_model.sh \
          "sweep-7b-$n" "checkpoints/q38-7b-$n" 1 Qwen/Qwen2.5-VL-7B-Instruct
    done
  fi
  cleanup_containers
  publish
}

# ---------------------------------------------------------------------------
# A2  one-shot regimes (GPU0), run in parallel with the sweep
# ---------------------------------------------------------------------------
stage_oneshot() {
  log "STAGE A2 one-shot regimes"
  [ -f data/clinocr/exemplars.jsonl ] || { log "no exemplars; run export_exemplars.py"; return 1; }
  export MAX_LEN=${MAX_LEN:-16384}
  for shot in homo hetero; do
    bash serve_and_eval.sh "oneshot-dotsmocr-$shot" rednote-hilab/dots.mocr 0 \
      data/clinocr "$shot" 8
    bash serve_and_eval.sh "oneshot-qwen25vl7b-$shot" Qwen/Qwen2.5-VL-7B-Instruct 0 \
      data/clinocr "$shot" 8
  done
  cleanup_containers
  publish
}

# ---------------------------------------------------------------------------
# A3  OmniDocBench generalisation (only if time remains)
# ---------------------------------------------------------------------------
stage_omnidoc() {
  [ "$(left)" -lt 5400 ] && { log "A3 skipped (not enough time)"; return 0; }
  log "STAGE A3 OmniDocBench"
  if [ ! -f data/omnidoc/eval.jsonl ]; then
    .venv/bin/python3 export_omnidocbench.py > logs/export_omnidoc.log 2>&1 \
      || { log "omnidoc export failed"; return 1; }
  fi
  # Qwen3.8 first (needs both GPUs); then the single-GPU models on GPU0.
  bash serve_and_eval.sh omnidoc-qwen38-27b Qwen/Qwen3.8-27B-FP8 0,1 data/omnidoc zero 8
  for m in "omnidoc-qwen25vl7b Qwen/Qwen2.5-VL-7B-Instruct" \
           "omnidoc-qwen25vl3b Qwen/Qwen2.5-VL-3B-Instruct" \
           "omnidoc-dots-mocr rednote-hilab/dots.mocr" \
           "omnidoc-olmocr-2 allenai/olmOCR-2-7B-1025"; do
    [ "$(left)" -lt 2400 ] && break
    set -- $m
    bash serve_and_eval.sh "$1" "$2" 0 data/omnidoc zero 8
  done
  cleanup_containers
  publish
}

# ---------------------------------------------------------------------------
# B3  runaway mitigation (only if time remains)
# ---------------------------------------------------------------------------
stage_runaway() {
  [ "$(left)" -lt 3600 ] && { log "B3 skipped (not enough time)"; return 0; }
  log "STAGE B3 runaway mitigation (no_repeat_ngram_size=6)"
  has "norepeat-qwen25vl7b" || \
    bash run_model.sh norepeat-qwen25vl7b Qwen/Qwen2.5-VL-7B-Instruct 0 "$PROMPT" \
      --no-repeat-ngram-size 6
  has "norepeat-paddleocr-vl" || \
    bash run_model.sh norepeat-paddleocr-vl PaddlePaddle/PaddleOCR-VL 1 \
      "OCR:"
  cleanup_containers
  publish
}

# ---------------------------------------------------------------------------
main() {
  log "overnight run 2 start; deadline $(date -d @$DEADLINE) ($(left)s)"
  stage_noise
  stage_mint
  # Minting held both GPUs (TP=2); make sure they are released before the two
  # lanes below claim one each. (waiting inside a lane would deadlock, since
  # the sibling lane legitimately keeps the other GPU busy.)
  wait_gpus_free
  # B2 (GPU1) and A2 (GPU0) are independent lanes -> run together
  stage_sweep > logs/lane_b2.log 2>&1 &
  local bs=$!
  stage_oneshot > logs/lane_a2.log 2>&1 &
  local os=$!
  wait "$bs" "$os"
  stage_omnidoc
  stage_runaway
  publish
  log "ALL DONE ($(left)s left)"
  echo "done $(date -u '+%Y-%m-%d %H:%M UTC')" > logs/overnight2.done
}

main "$@"
