#!/usr/bin/env python3
"""Score a predictions.jsonl against a ground-truth jsonl (doc_id + ground_truth).

Usage: python3 score_preds.py --preds results/x/predictions.jsonl \
    --gt data/clinocr/eval.jsonl --out results/x [--wall SECONDS]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cli import cer, normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--wall", type=float, default=None)
    args = ap.parse_args()

    gt = {json.loads(l)["doc_id"]: json.loads(l) for l in open(args.gt)}
    per_doc, per_subset = [], {}
    lat = []
    n_err = 0
    for line in open(args.preds):
        r = json.loads(line)
        g = gt.get(r["doc_id"])
        if g is None:
            continue
        c = cer(normalize(g["ground_truth"]), normalize(r.get("prediction") or ""))
        per_subset.setdefault(g.get("subset", "?"), []).append(c)
        per_doc.append({"doc_id": r["doc_id"], "subset": g.get("subset"),
                        "cer": round(c, 4), "error": r.get("error")})
        if r.get("error"):
            n_err += 1
        if r.get("latency_s"):
            lat.append(r["latency_s"])

    cers = [d["cer"] for d in per_doc]
    # A page the model cannot read can send it into a repetition loop, which
    # pins that document at the CER cap of 2.0. Which pages do that varies
    # between identical runs (continuous batching is not deterministic), and
    # those few documents dominate the mean: two runs of the same adapter at
    # the same config gave mean CER 0.2844 and 0.3072 while their medians were
    # 0.0944 and 0.0954. Report the stable statistics alongside the mean.
    runaway = [c for c in cers if c >= 2.0]
    kept = [c for c in cers if c < 2.0]
    wall = args.wall or (sum(lat) if lat else 1)
    summary = {
        "model": args.name or os.path.basename(args.out.rstrip("/")),
        "n_docs": len(per_doc),
        "n_errors": n_err,
        "mean_cer": round(sum(cers) / len(cers), 4),
        "median_cer": round(sorted(cers)[len(cers) // 2], 4),
        "n_runaway": len(runaway),
        "mean_cer_excl_runaway": round(sum(kept) / len(kept), 4) if kept else None,
        "per_subset": {s: {"mean_cer": round(sum(v) / len(v), 4), "n": len(v)}
                       for s, v in sorted(per_subset.items())},
        "wall_time_s": round(wall, 1),
        "pages_per_sec": round(len(per_doc) / wall, 3),
        "avg_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump(per_doc, open(os.path.join(args.out, "per_doc.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
