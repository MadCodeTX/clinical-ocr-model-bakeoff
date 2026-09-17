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


def _image_content(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _maybe_abs(path, base):
    if os.path.isabs(path) or os.path.exists(path):
        return path
    cand = os.path.join(base, path)
    return cand if os.path.exists(cand) else path


def load_exemplars(path, base):
    """Map doc_id -> {image, ground_truth} for the one-shot exemplar pool."""
    ex = {}
    if not path or not os.path.exists(path):
        return ex
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ex[r["doc_id"]] = {
                "image": _maybe_abs(r["image"], base),
                "ground_truth": r["ground_truth"],
            }
    return ex


def build_messages(item, prompt, shot, exemplars, base):
    """Zero-shot, or one-shot with a homogeneous/heterogeneous exemplar.

    One-shot prepends a clean exemplar page + its ground truth as a
    prior user/assistant turn, then poses the same prompt on the query page.
    homo_id is the same-template exemplar, hetero_id a different-template one.
    """
    messages = []
    if shot in ("homo", "hetero") and exemplars:
        key = "homo_id" if shot == "homo" else "hetero_id"
        ex = exemplars.get(item.get(key))
        if ex:
            messages.append({"role": "user", "content": [
                _image_content(_maybe_abs(ex["image"], base)),
                {"type": "text", "text": prompt}]})
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": ex["ground_truth"]}]})
    messages.append({"role": "user", "content": [
        _image_content(item["image"]),
        {"type": "text", "text": prompt}]})
    return messages


def predict_one(endpoint, model, prompt, max_tokens, item, repetition_penalty=None,
                shot="zero", exemplars=None, base=None, no_repeat_ngram=None):
    payload = {
        "model": model,
        "messages": build_messages(item, prompt, shot, exemplars, base or "."),
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    # Greedy decoding on a page the model cannot read degenerates into a loop
    # (" D. D. D. D. ..."), which pins CER at the 2.0 cap. vLLM accepts
    # repetition_penalty as an OpenAI-API extension.
    if repetition_penalty:
        payload["repetition_penalty"] = repetition_penalty
    # A blocked n-gram stops exact repetition loops (" D. D. D.") without
    # penalising legitimately repeated tokens; vLLM/OpenAI extension.
    if no_repeat_ngram:
        payload["no_repeat_ngram_size"] = no_repeat_ngram
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
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="e.g. 1.05; suppresses runaway repetition on unreadable pages")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=None,
                    help="e.g. 6; blocks exact n-gram loops without penalising repeated tokens")
    ap.add_argument("--shot", choices=["zero", "homo", "hetero"], default="zero",
                    help="one-shot regime: prepend a homogeneous/heterogeneous exemplar")
    ap.add_argument("--exemplars", default=None,
                    help="exemplars.jsonl (defaults to data/clinocr/exemplars.jsonl)")
    ap.add_argument("--resume", action="store_true",
                    help="keep existing predictions.jsonl and skip finished docs")
    args = ap.parse_args()

    with open(os.path.join(args.data, "eval.jsonl")) as f:
        items = [json.loads(l) for l in f]

    exemplars = {}
    if args.shot != "zero":
        ex_path = args.exemplars or os.path.join(args.data, "exemplars.jsonl")
        exemplars = load_exemplars(ex_path, os.path.dirname(os.path.abspath(args.data)))
        n_missing = sum(1 for it in items if not exemplars.get(
            it.get("homo_id" if args.shot == "homo" else "hetero_id")))
        if n_missing:
            print(f"WARNING: {n_missing}/{len(items)} docs have no {args.shot} exemplar")
        print(f"{args.shot}-shot: loaded {len(exemplars)} exemplars from {ex_path}")

    os.makedirs(args.out, exist_ok=True)
    preds_path = os.path.join(args.out, "predictions.jsonl")

    # Long runs get killed by time budgets; --resume keeps the docs already on
    # disk (one flushed JSON line each) instead of restarting the whole eval.
    done_ids = set()
    if args.resume and os.path.exists(preds_path):
        with open(preds_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["doc_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        items = [it for it in items if it["doc_id"] not in done_ids]
        print(f"resume: {len(done_ids)} already done, {len(items)} remaining")

    print(f"{args.name}: {len(items)} documents, endpoint={args.endpoint}")
    base = os.path.dirname(os.path.abspath(args.data))
    t_start = time.time()
    done = 0
    mode = "a" if (args.resume and done_ids) else "w"
    with open(preds_path, mode) as out_f, ThreadPoolExecutor(args.concurrency) as ex:
        futures = {ex.submit(predict_one, args.endpoint, args.model,
                             args.prompt, args.max_tokens, it,
                             args.repetition_penalty, args.shot, exemplars,
                             base, args.no_repeat_ngram_size): it for it in items}
        for fut, it in futures.items():
            res = fut.result()
            out_f.write(json.dumps(res) + "\n")
            out_f.flush()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(items)} done")
    wall = time.time() - t_start

    # score (read GT from the full eval set -- on resume `items` is a subset)
    with open(os.path.join(args.data, "eval.jsonl")) as f:
        gt_by_id = {json.loads(l)["doc_id"]: json.loads(l) for l in f}
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
    # Runaway-robust stats: a page the model cannot read can pin CER at the 2.0
    # cap, and which pages do that varies between identical runs. Rank on
    # median; report n_runaway and mean_cer_excl_runaway alongside the mean.
    runaway = [c for c in all_cers if c >= 2.0]
    kept = [c for c in all_cers if c < 2.0]
    summary = {
        "model": args.name,
        "shot": args.shot,
        "n_docs": len(per_doc),
        "n_errors": n_err,
        "mean_cer": round(sum(all_cers) / len(all_cers), 4),
        "median_cer": round(sorted(all_cers)[len(all_cers) // 2], 4),
        "n_runaway": len(runaway),
        "mean_cer_excl_runaway": round(sum(kept) / len(kept), 4) if kept else None,
        "per_subset": subset_stats,
        "wall_time_s": round(wall, 1),
        "pages_per_sec": round(len(per_doc) / wall, 3),
        "avg_latency_s": round(sum(lat) / max(len(lat), 1), 2),
    }
    with open(os.path.join(args.out, "per_doc.json"), "w") as f:
        json.dump(per_doc, f, indent=1)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
