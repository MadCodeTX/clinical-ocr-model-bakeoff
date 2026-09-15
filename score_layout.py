#!/usr/bin/env python3
"""Score dots.mocr / dots.ocr *layout-prompt* output.

The layout prompt returns a single JSON object: one entry per layout element
with a bbox, a category and the region's text (Markdown / LaTeX / HTML). Plain
CER on the raw JSON is meaningless (the coordinates and category names are not
document text), so this:

  * parses the JSON and concatenates the text fields in reading order, then
    computes CER against the ground truth -- comparable with the plain-prompt runs;
  * reports coordinate coverage (how many elements carry a valid [x1,y1,x2,y2]);
  * reports the category mix and how often a Table element's HTML survives;
  * measures whether selection controls (checked boxes / X marks) survive, which
    is the whole point of trying this prompt.

Usage: python3 score_layout.py --name layout-dots-mocr --data data/clinocr
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cli import cer, normalize  # noqa: E402

BOX = r"[☐☒☑]"
CHECKED = r"[☒☑]"
XLINE = re.compile(r"(?m)^\s*[\*_\s]*[Xx][\*_\s]*$")
TICK = r"(✓|✔|\[x\]|\[X\])"


def parse_layout(text):
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z0-9]*\s*\n", "", t)
    t = re.sub(r"\n```\s*$", "", t)
    obj = None
    for cand in (t, t[t.find("{"):t.rfind("}") + 1] if "{" in t and "}" in t else t):
        try:
            obj = json.loads(cand)
            break
        except Exception:
            obj = None
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k in ("layout", "layout_dets", "elements", "blocks", "pages", "result", "data"):
            if isinstance(obj.get(k), list):
                return obj[k]
        if "bbox" in obj:
            return [obj]
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return []
    return obj if isinstance(obj, list) else None


def block_text(b):
    for k in ("text", "markdown", "content"):
        if b.get(k):
            return str(b[k])
    for k in ("html", "latex"):
        if b.get(k):
            return str(b[k])
    return ""


def bbox_of(b):
    for k in ("bbox", "box", "bounds", "poly"):
        v = b.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            return v
    return None


def is_table(b):
    c = str(b.get("category") or b.get("category_type") or b.get("label") or "").lower()
    return "table" in c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/clinocr")
    args = ap.parse_args()

    gt = {json.loads(l)["doc_id"]: json.loads(l)
          for l in open(os.path.join(args.data, "eval.jsonl"))}
    preds = [json.loads(l) for l in
             open(os.path.join("results", args.name, "predictions.jsonl"))]

    parsed = bbox_ok = blocks_total = tables = table_html = 0
    cats = Counter()
    cers = []
    box_docs = [d for d in gt.values() if re.search(BOX, d["ground_truth"])]
    box_ids = {d["doc_id"] for d in box_docs}
    checked_gt = sum(len(re.findall(CHECKED, d["ground_truth"])) for d in box_docs)
    checked_pr = 0
    x_ids = {d["doc_id"] for d in gt.values()
             if XLINE.search(d["ground_truth"]) and not re.search(BOX, d["ground_truth"])}
    x_gt = sum(len(XLINE.findall(gt[d]["ground_truth"])) for d in x_ids)
    x_pr = 0

    for r in preds:
        doc = r["doc_id"]
        g = gt.get(doc)
        if not g:
            continue
        blocks = parse_layout(r.get("prediction"))
        if blocks is None:
            cers.append(1.0)
            continue
        parsed += 1
        cat_text_parts = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            blocks_total += 1
            if bbox_of(b):
                bbox_ok += 1
            c = str(b.get("category") or b.get("category_type") or b.get("label") or "?")
            cats[c] += 1
            if is_table(b):
                tables += 1
                if b.get("html") or "<table" in str(b.get("text", "")).lower():
                    table_html += 1
            cat_text_parts.append(block_text(b))
        joined = "\n".join(cat_text_parts)
        cers.append(cer(normalize(g["ground_truth"]), normalize(joined)))
        if doc in box_ids:
            checked_pr += min(len(re.findall(CHECKED, joined)),
                              len(re.findall(CHECKED, g["ground_truth"])))
        if doc in x_ids:
            x_pr += min(len(XLINE.findall(joined)) + len(re.findall(TICK, joined)),
                        len(XLINE.findall(g["ground_truth"])))

    n = len(preds)
    cers.sort()
    out = {
        "model": args.name,
        "n_docs": n,
        "json_parse_rate": round(parsed / n, 3),
        "elements_total": blocks_total,
        "elements_with_bbox": bbox_ok,
        "bbox_rate": round(bbox_ok / max(blocks_total, 1), 3),
        "tables": tables,
        "tables_with_html": table_html,
        "text_cer_median": round(cers[len(cers) // 2], 4) if cers else None,
        "text_cer_mean": round(sum(cers) / len(cers), 4) if cers else None,
        "checked_mark_recall": round(checked_pr / checked_gt, 3) if checked_gt else None,
        "x_mark_recall": round(x_pr / x_gt, 3) if x_gt else None,
        "categories": dict(cats.most_common()),
    }
    json.dump(out, open(os.path.join("results", args.name, "layout_score.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()