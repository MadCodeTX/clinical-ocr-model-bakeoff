#!/usr/bin/env python3
"""Turn a teacher's predictions over the synthetic corpus into a training set.

This is the "no ground truth" path: the teacher reads unlabeled scans and its
transcripts become the student's labels. e13 did this inline for one teacher;
now that we compare several, it is a script.

Usage:
  python3 build_teacher_set.py --preds results/teacher-labels-qwen38/predictions.jsonl \
      --out data/synth/teacher_labels_qwen38.jsonl
"""
import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))

# Reasoning teachers (Qwen3.8) default to emitting a  thinking...<｜end▁of▁thinking｜> preamble
# before the transcript. vLLM's reasoning handling often leaves only the closing
# tag in the content, so strip everything up to and including the last closer.
# Left in, the student learns to emit it too and label noise jumps from ~0.01
# CER to ~0.40, so strip unless asked not to.
REASONING_CLOSED = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
REASONING_PREFIX = re.compile(r"^.*(?:</think\s*>|<｜end▁of▁thinking｜>)", re.DOTALL | re.IGNORECASE)
REASONING_OPEN = re.compile(r"^.*?<think\b[^>]*>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text):
    text = REASONING_CLOSED.sub("", text)
    # greedy .* removes everything up to the LAST closer (handles a bare
    # '</think>' with no opening tag, which is what vLLM often returns)
    text = REASONING_PREFIX.sub("", text)
    if re.search(r"<think\b", text, re.IGNORECASE):
        text = REASONING_OPEN.sub("", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="teacher predictions.jsonl")
    ap.add_argument("--labels", default=os.path.join(ROOT, "data", "synth", "labels.jsonl"),
                    help="synthetic corpus manifest (image paths + exact text)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=32,
                    help="drop transcripts shorter than this; a teacher that "
                         "returned almost nothing teaches the student to do the same")
    ap.add_argument("--keep-reasoning", action="store_true",
                    help="keep a teacher's  thinking...<｜end▁of▁thinking｜> preamble (default: strip it)")
    args = ap.parse_args()

    preds = {}
    for line in open(args.preds):
        r = json.loads(line)
        preds[r["doc_id"]] = r

    rows = [json.loads(l) for l in open(args.labels)]
    kept = dropped_empty = dropped_short = dropped_error = 0
    stripped = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            p = preds.get(r["id"])
            if not p:
                continue
            if p.get("error"):
                dropped_error += 1
                continue
            raw = (p.get("prediction") or "")
            text = raw if args.keep_reasoning else strip_reasoning(raw)
            if text != raw.strip():
                stripped += 1
            text = text.strip()
            if not text:
                dropped_empty += 1
                continue
            if len(text) < args.min_chars:
                dropped_short += 1
                continue
            f.write(json.dumps({"id": r["id"], "image": r["image"],
                                "text": text, "handwriting": r["handwriting"]}) + "\n")
            kept += 1

    print(f"wrote {args.out}")
    print(f"  kept            {kept}")
    print(f"  stripped reasoning {stripped}")
    print(f"  dropped empty   {dropped_empty}")
    print(f"  dropped short   {dropped_short}  (< {args.min_chars} chars)")
    print(f"  dropped errored {dropped_error}")
    if kept == 0:
        raise SystemExit("no usable teacher labels")


if __name__ == "__main__":
    main()
