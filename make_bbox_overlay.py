#!/usr/bin/env python3
"""Render a document image with the layout model's bounding boxes overlaid.

Produces two PNGs: the raw input page, and the same page with each predicted
region outlined and labelled by category. Used for the slide deck.

Usage: python3 make_bbox_overlay.py --layout results/layout-dots-mocr/predictions.jsonl \
    --doc-id tables_t16_s2 --data data/clinocr --out docs/deck
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_layout import parse_layout, bbox_of  # noqa: E402

COLORS = {
    "Text": (31, 119, 180),
    "Section-header": (214, 39, 40),
    "Title": (148, 103, 189),
    "Page-header": (140, 86, 75),
    "Page-footer": (127, 127, 127),
    "List-item": (44, 160, 44),
    "Table": (255, 127, 14),
    "Picture": (23, 190, 207),
    "Caption": (188, 189, 34),
    "Footnote": (227, 119, 194),
    "Formula": (148, 103, 189),
}
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]


def font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def poly_to_xyxy(b):
    v = bbox_of(b)
    if not v or len(v) < 4:
        return None
    if len(v) == 4:
        x1, y1, x2, y2 = v
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    xs, ys = v[0::2], v[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--data", default="data/clinocr")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = {json.loads(l)["doc_id"]: json.loads(l)
          for l in open(os.path.join(args.data, "eval.jsonl"))}
    pred = {json.loads(l)["doc_id"]: json.loads(l)["prediction"]
            for l in open(args.layout)}
    doc = gt[args.doc_id]
    img = Image.open(doc["image"]).convert("RGB")

    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, args.doc_id)
    img.save(base + "_input.png")

    W, H = img.size
    # draw on a 2x canvas so thin boxes/labels survive slide scaling
    scale = max(1, int(1600 / max(W, H)) + 1)
    big = img.resize((W * scale, H * scale), Image.LANCZOS)
    d = ImageDraw.Draw(big)
    f = font(max(14, 10 * scale))
    blocks = parse_layout(pred[args.doc_id]) or []
    counts = {}
    for b in blocks:
        xy = poly_to_xyxy(b)
        if not xy:
            continue
        cat = str(b.get("category") or b.get("category_type") or "?")
        counts[cat] = counts.get(cat, 0) + 1
        col = COLORS.get(cat, (200, 0, 200))
        x1, y1, x2, y2 = [c * scale for c in xy]
        d.rectangle([x1, y1, x2, y2], outline=col, width=max(2, scale))
        label = cat
        tb = d.textbbox((x1, y1), label, font=f)
        d.rectangle([tb[0], tb[1], tb[2] + 4, tb[3] + 2], fill=col)
        d.text((x1 + 2, y1), label, fill=(255, 255, 255), font=f)
    big.save(base + "_bbox.png")
    print(f"{args.doc_id}: {len(blocks)} blocks, categories={counts}")
    print("  ->", base + "_input.png", "and", base + "_bbox.png")


if __name__ == "__main__":
    main()