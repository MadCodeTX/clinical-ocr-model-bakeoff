#!/usr/bin/env python3
"""Eval client: send document images to an OpenAI-compatible vLLM endpoint,
collect predictions, compute CER vs ground truth.

Usage:
  python3 eval_cli.py --endpoint http://localhost:8001 --model ocr \
      --name granite-docling --data data/clinocr --out results/granite-docling
"""
import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz.distance import Levenshtein

MD_CHARS = set("#*_`|>[]~")


def normalize(t: str) -> str:
    """Normalize model output / ground truth for CER: lowercase, strip markdown
    and XML-ish (DocTags) markup, collapse whitespace."""
    t = t.lower()
    t = re.sub(r"<[^>]{1,40}>", " ", t)          # doctags / xml elements
    t = "".join(ch if ch not in MD_CHARS else " " for ch in t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0
    d = Levenshtein.distance(ref, hyp)
    return min(d / len(ref), 2.0)


def predict_one(endpoint, model, prompt, max_tokens, item):
    with open(item["image"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        text = resp["choices"][0]["message"]["content"] or ""
        usage = resp.get("usage", {})
        err = None
    except Exception as e:  # noqa: BLE001
        text, usage, err = "", {}, repr(e)
    return {
        "doc_id": item["doc_id"],
        "subset": item["subset"],
        "latency_s": round(time.time() - t0, 2),
        "completion_tokens": usage.get("completion_tokens"),
        "error": err,
        "prediction": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--name", required=True, help="label for results")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clinocr"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    with open(os.path.join(args.data, "eval.jsonl")) as f:
        items = [json.loads(l) for l in f]
    print(f"{args.name}: {len(items)} documents, endpoint={args.endpoint}")

    os.makedirs(args.out, exist_ok=True)
    preds_path = os.path.join(args.out, "predictions.jsonl")

    t_start = time.time()
    done = 0
    with open(preds_path, "w") as out_f, ThreadPoolExecutor(args.concurrency) as ex:
        futures = {ex.submit(predict_one, args.endpoint, args.model,
                             args.prompt, args.max_tokens, it): it for it in items}
        for fut in futures:
            pass
        for fut, it in futures.items():
            res = fut.result()
            out_f.write(json.dumps(res) + "\n")
            out_f.flush()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(items)} done")
    wall = time.time() - t_start

    # score
    gt_by_id = {it["doc_id"]: it for it in items}
    per_doc, per_subset = [], {}
    with open(preds_path) as f:
        for line in f:
            r = json.loads(line)
            ref = normalize(gt_by_id[r["doc_id"]]["ground_truth"])
            hyp = normalize(r["prediction"])
            c = cer(ref, hyp)
            per_doc.append({"doc_id": r["doc_id"], "subset": r["subset"],
                            "cer": round(c, 4), "error": r["error"]})
            per_subset.setdefault(r["subset"], []).append(c)

    subset_stats = {s: {"mean_cer": round(sum(v) / len(v), 4), "n": len(v)}
                    for s, v in sorted(per_subset.items())}
    all_cers = [d["cer"] for d in per_doc]
    with open(preds_path) as f:
        lat = [json.loads(l)["latency_s"] for l in f]
    n_err = sum(1 for d in per_doc if d["error"])
    summary = {
        "model": args.name,
        "n_docs": len(items),
        "n_errors": n_err,
        "mean_cer": round(sum(all_cers) / len(all_cers), 4),
        "median_cer": round(sorted(all_cers)[len(all_cers) // 2], 4),
        "per_subset": subset_stats,
        "wall_time_s": round(wall, 1),
        "pages_per_sec": round(len(items) / wall, 3),
        "avg_latency_s": round(sum(lat) / max(len(lat), 1), 2),
    }
    with open(os.path.join(args.out, "per_doc.json"), "w") as f:
        json.dump(per_doc, f, indent=1)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
