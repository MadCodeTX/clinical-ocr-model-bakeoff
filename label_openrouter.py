#!/usr/bin/env python3
"""Label page images with a vision LLM through OpenRouter.

Produces the full-page transcription that our distillation training expects
(prompt: "Extract the text content from this image."). Writes predictions.jsonl
with {doc_id, prediction, latency_s, completion_tokens, cost, error}.

Resumable: --resume skips doc_ids already present. The API key is read from the
pi auth store (~/.pi/agent/auth.json) unless OPENROUTER_API_KEY is set.

Usage:
  python3 label_openrouter.py --manifest data/medreal/manifest.jsonl \
     --out results/medreal-deepseek --model deepseek/deepseek-v4.1-flash \
     --concurrency 8 --resume
"""
import argparse
import base64
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT = "Extract the text content from this image."
SYSTEM = ("You are a document OCR engine. Transcribe every visible character of "
          "the document, preserving reading order, tables and layout. Output ONLY "
          "the raw transcription: no introduction, no commentary, no markdown code "
          "fences, no explanations.")


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    return json.load(open(os.path.expanduser("~/.pi/agent/auth.json")))["openrouter"]["key"]


PREAMBLE = re.compile(
    r"(?is)^\s*(here(?:'s| is)|sure[,!]?|below is|the following is|"
    r"i (?:have|will|'ll)|certainly|extracted text|transcription:)\b.*?\n\s*\n")


def clean(text):
    text = re.sub(r"(?s)^\s*```[a-zA-Z0-9]*\s*\n", "", text)
    text = re.sub(r"(?s)\n```\s*$", "", text)
    # drop a leading "Here is ..." paragraph if one survived the system prompt
    m = PREAMBLE.match(text)
    if m:
        text = text[m.end():]
    return text.strip()


def label_one(endpoint, model, key, item, max_tokens, retries=3):
    path = item["image"] if os.path.isabs(item["image"]) else os.path.join(ROOT, item["image"])
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT}]},
        ],
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "HTTP-Referer": "https://localhost/ocr-bench",
                 "X-Title": "clinical-ocr-bench"})
    t0 = time.time()
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
            usage = resp.get("usage", {})
            return {
                "doc_id": item["doc_id"],
                "prediction": clean(resp["choices"][0]["message"]["content"] or ""),
                "latency_s": round(time.time() - t0, 2),
                "completion_tokens": usage.get("completion_tokens"),
                "cost": usage.get("cost"),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            time.sleep(2 * (attempt + 1))
    return {"doc_id": item["doc_id"], "prediction": "", "latency_s": round(time.time() - t0, 2),
            "completion_tokens": None, "cost": None, "error": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "medreal", "manifest.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="https://openrouter.ai/api")
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    key = api_key()
    items = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        items = items[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    preds_path = os.path.join(args.out, "predictions.jsonl")
    done = set()
    if args.resume and os.path.exists(preds_path):
        for line in open(preds_path):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
    todo = [it for it in items if it["doc_id"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to label via {args.model}")

    t0 = time.time()
    mode = "a" if done else "w"
    n = 0
    with open(preds_path, mode) as out, ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(label_one, args.endpoint, args.model, key, it,
                          args.max_tokens): it for it in todo}
        for fu in futs:
            r = fu.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

    # summary over the whole file
    rows = [json.loads(l) for l in open(preds_path)]
    errs = sum(1 for r in rows if r["error"])
    cost = sum(r["cost"] or 0 for r in rows)
    lens = sorted(len(r["prediction"]) for r in rows)
    summary = {
        "model": args.model,
        "n": len(rows),
        "n_errors": errs,
        "total_cost_usd": round(cost, 4),
        "cost_per_page": round(cost / max(len(rows), 1), 5),
        "median_chars": lens[len(lens) // 2] if lens else 0,
        "wall_time_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()