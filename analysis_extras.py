#!/usr/bin/env python3
"""Extra analyses beyond headline CER:
  * per-template breakdown (which *document types* break which model)
  * CER vs image-quality correlates (blur, skew, resolution, ink)
  * failure taxonomy (empty output, repetition, truncation, length ratio)
  * worst-document inspection sheet
Writes reports/per_template.csv, reports/analysis.json, reports/worst_docs.md.
"""
import glob
import json
import os
import re

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")


def main():
    os.makedirs(REPORTS, exist_ok=True)
    gt = {json.loads(l)["doc_id"]: json.loads(l)
          for l in open(os.path.join(ROOT, "data", "clinocr", "eval.jsonl"))}
    models = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", "per_doc.json"))):
        name = os.path.basename(os.path.dirname(p))
        models[name] = {d["doc_id"]: d["cer"] for d in json.load(open(p))}
    if not models:
        print("no per_doc.json yet")
        return

    templates = sorted({g["template"] for g in gt.values()})
    out = {"per_template": {}, "taxonomy": {}, "correlates": {}}

    # ---- per-template -------------------------------------------------
    rows = []
    for name, cers in models.items():
        row = {"model": name}
        for t in templates:
            ids = [i for i in cers if gt.get(i, {}).get("template") == t]
            row[f"t{t}"] = round(float(np.mean([cers[i] for i in ids])), 4) if ids else None
        rows.append(row)
    with open(os.path.join(REPORTS, "per_template.csv"), "w") as f:
        cols = ["model"] + [f"t{t}" for t in templates]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    out["per_template"] = rows
    print("per-template CER written (reports/per_template.csv)")

    # ---- taxonomy -----------------------------------------------------
    for name in models:
        pp = os.path.join(ROOT, "results", name, "predictions.jsonl")
        if not os.path.exists(pp):
            continue
        preds = [json.loads(l) for l in open(pp)]
        empty = rep = trunc = 0
        ratios = []
        for r in preds:
            p = (r.get("prediction") or "").strip()
            g = gt.get(r["doc_id"])
            if not p:
                empty += 1
                continue
            toks = p.split()
            if len(toks) > 20:
                grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 4)]
                _, counts = np.unique(grams, return_counts=True)
                rep += int(counts.max() / len(grams) > 0.4)
            if p[-1].isalnum() and len(p) > 40 and not p.endswith((".", ")", "\"", ":")):
                trunc += 1
            if g:
                ratios.append(len(p) / max(len(g["ground_truth"]), 1))
        out["taxonomy"][name] = {
            "n": len(preds), "empty": empty, "repetition": rep,
            "suspected_truncation": trunc,
            "median_len_ratio": round(float(np.median(ratios)), 3) if ratios else None,
        }
    print(json.dumps(out["taxonomy"], indent=1))

    # ---- image-quality correlates (using the fastest model as reference) --
    ref = "paddleocr-vl" if "paddleocr-vl" in models else list(models)[0]
    feats = {}
    for doc_id, c in models[ref].items():
        p = os.path.join(ROOT, gt[doc_id]["image"])
        im = Image.open(p).convert("L")
        w, h = im.size
        a = np.asarray(im.resize((400, max(50, int(400 * h / w)))), dtype=np.float32)
        lap = (-4 * a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
               + np.roll(a, 1, 1) + np.roll(a, -1, 1)).var()
        feats[doc_id] = {"blurvar": float(lap), "mpx": w * h / 1e6,
                         "ink": float((a < 128).mean()), "w": w}
    keys = ["blurvar", "mpx", "ink", "w"]
    corr = {}
    for k in keys:
        xs = np.array([feats[i][k] for i in models[ref]])
        ys = np.array([models[ref][i] for i in models[ref]])
        corr[k] = round(float(np.corrcoef(xs, ys)[0, 1]), 3) if xs.std() > 0 else None
    out["correlates"] = {"reference_model": ref, "pearson_cer_vs": corr}
    print("CER correlates for", ref, corr)

    # ---- worst documents ---------------------------------------------
    best = min(models, key=lambda m: float(np.mean(list(models[m].values()))))
    worst = sorted(models[best], key=lambda i: -models[best][i])[:15]
    lines = [f"# Worst documents for `{best}`\n",
             "The 15 highest-CER pages. Useful for deciding what a router should "
             "escalate and what data a student needs.\n"]
    preds = {}
    pp = os.path.join(ROOT, "results", best, "predictions.jsonl")
    if os.path.exists(pp):
        preds = {json.loads(l)["doc_id"]: json.loads(l) for l in open(pp)}
    for doc_id in worst:
        g = gt[doc_id]
        lines.append(f"\n## `{doc_id}` — subset **{g['subset']}**, template {g['template']}, "
                     f"CER {models[best][doc_id]:.2f}\n")
        lines.append(f"![{doc_id}](../{g['image']})\n")
        lines.append("**ground truth (excerpt)**\n```\n" +
                     g["ground_truth"][:600].replace("\n", " ⏎ ") + "\n```\n")
        pr = (preds.get(doc_id, {}).get("prediction") or "")[:600]
        lines.append("**prediction (excerpt)**\n```\n" + pr.replace("\n", " ⏎ ") + "\n```\n")
    with open(os.path.join(REPORTS, "worst_docs.md"), "w") as f:
        f.write("\n".join(lines))
    print("worst documents written (reports/worst_docs.md)")

    with open(os.path.join(REPORTS, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
