#!/usr/bin/env python3
"""Combine per-model summaries into one report table."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", "summary.json"))):
    s = json.load(open(p))
    rows.append(s)

rows.sort(key=lambda r: r["mean_cer"])
print(f"{'model':<18}{'CER':>8}{'med':>8}{'pages/s':>9}  per-subset CER (n/hw/poor/rot/tbl/mix)")
print("-" * 100)
for r in rows:
    ps = r["per_subset"]
    order = ["normal", "handwriting", "poor", "rotated", "tables", "mixed"]
    cells = " ".join(f"{ps[k]['mean_cer']:.3f}" if k in ps else "  n/a" for k in order)
    print(f"{r['model']:<18}{r['mean_cer']:>8.4f}{r['median_cer']:>8.4f}"
          f"{r['pages_per_sec']:>9.2f}  {cells}")

# implied cost math vs Azure Layout ($0.01/page)
print("\nvs Azure Layout Layout-OCR @ $0.01/page (electricity ~$0.15/kWh, ~450W/GPU load):")
for r in rows:
    if r["model"] == "tesseract":
        continue
    daily_pages = r["pages_per_sec"] * 86400
    kwh = 0.45 * 24
    cost = kwh * 0.15
    layout_cost = daily_pages * 0.01
    print(f"  {r['model']:<18} {daily_pages:>10,.0f} pages/day on one 4090"
          f"  -> ~${cost:.2f}/day electricity vs ${layout_cost:,.0f}/day Azure Layout")
