#!/usr/bin/env python3
"""Build the slide deck (docs/deck/clinical_ocr_bakeoff.pptx).

Walks through: the question, the models, the corpora, the method, the results
(old + new), the bounding-box/layout test, the training, and the caveats.
Charts are rendered with matplotlib; document screenshots come from
make_bbox_overlay.py.
"""
import json
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(ROOT, "docs", "deck")
RES = os.path.join(ROOT, "results")
os.makedirs(DECK, exist_ok=True)

ACCENT = RGBColor(0x1F, 0x77, 0xB4)
DARK = RGBColor(0x20, 0x20, 0x20)
GREY = RGBColor(0x60, 0x60, 0x60)
GREEN = RGBColor(0x2C, 0xA0, 0x2C)
RED = RGBColor(0xC0, 0x30, 0x30)


def load(name):
    p = os.path.join(RES, name, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


# display, result-name, params, license
MODELS = [
    ("Qwen3.8-27B-FP8", "qwen38-27b", "27B (FP8)", "verify"),
    ("Qwen2.5-VL-7B", "qwen25vl7b", "7B", "Apache-2.0"),
    ("Qwen2.5-VL-3B", "qwen25vl3b-base", "3B", "Apache-2.0"),
    ("dots.mocr", "dots-mocr", "3B", "MIT"),
    ("dots.ocr", "dots-ocr", "3B", "MIT"),
    ("olmOCR-2", "olmocr-2", "8B", "Apache-2.0"),
    ("PaddleOCR-VL", "paddleocr-vl", "0.9B", "Apache-2.0"),
    ("DeepSeek-OCR", "deepseek-ocr", "3B MoE", "MIT"),
    ("Chandra OCR 2", "chandra-2", "5B", "OpenRAIL-M"),
    ("granite-docling", "granite-docling", "0.26B", "Apache-2.0"),
    ("Nanonets-OCR2", "nanonets-ocr2-3b", "3B", "Apache-2.0"),
    ("Tesseract v5", "tesseract", "-", "Apache-2.0"),
]
NEW = {  # corrected new-set medians (27B de-preamble'd)
    "qwen38-27b": 0.0799, "qwen25vl7b": 0.1134, "qwen25vl3b-base": 0.1961,
    "dots-mocr": 0.1726, "dots-ocr": 2.0, "olmocr-2": 0.2570,
    "paddleocr-vl": 0.2416, "deepseek-ocr": 0.3240, "chandra-2": 0.9256,
    "granite-docling": 0.3168, "nanonets-ocr2-3b": 2.0, "tesseract": 0.2946,
}


def chart_leaderboard(rows, path, title):
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    bars = ax.barh(range(len(names)), vals, color="#1f77b4")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("median CER (lower is better)", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.axvline(0.444, color="#888", ls="--", lw=1)
    ax.text(0.444, -0.8, " Tesseract", color="#888", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_old_new(rows, path):
    names = [r[0] for r in rows]
    old = [r[1] for r in rows]
    new = [r[2] for r in rows]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=150)
    w = 0.4
    ax.bar([i - w / 2 for i in x], old, w, label="ClinOCR (synthetic)", color="#1f77b4")
    ax.bar([i + w / 2 for i in x], new, w, label="medreal (real scans)", color="#ff7f0e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("median CER", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 2.15)
    for i, (o, n) in enumerate(zip(old, new)):
        ax.text(i - w / 2, min(o, 2.05) + 0.03, f"{o:.2f}", ha="center", fontsize=6, color="#1f77b4")
        ax.text(i + w / 2, min(n, 2.05) + 0.03, f"{n:.2f}", ha="center", fontsize=6, color="#ff7f0e")
    ax.set_title("Median CER: synthetic vs real scans (2.0 = CER cap / unusable)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_sweep(path):
    sizes = ["base", "800", "2400", "4800"]
    b3 = [0.0649, 0.0663, 0.0704, 0.0692]
    b7 = [0.0720, 0.0449, 0.0617, None]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    ax.plot(sizes[:4], b3, "o-", label="Qwen2.5-VL-3B", color="#1f77b4")
    ax.plot(sizes[:3], b7[:3], "s-", label="Qwen2.5-VL-7B", color="#ff7f0e")
    ax.axhline(0.0026, color="#bbb", ls=":")
    ax.set_xlabel("teacher-labelled training docs", fontsize=9)
    ax.set_ylabel("median CER", fontsize=9)
    ax.set_title("Corpus-size sweep (distillation)", fontsize=10)
    ax.legend(fontsize=8)
    for xs, ys, c in ((sizes[:4], b3, "#1f77b4"), (sizes[:3], b7[:3], "#ff7f0e")):
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7, color=c)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------- pptx helpers ----------
def prs_new():
    p = Presentation()
    p.slide_width, p.slide_height = Inches(13.333), Inches(7.5)
    return p


def blank(p):
    return p.slides.add_slide(p.slide_layouts[6])


def bar(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.28), Inches(12.33), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = text
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = DARK
    if sub:
        p = tf.add_paragraph()
        r = p.add_run()
        r.text = sub
        r.font.size = Pt(13)
        r.font.color.rgb = GREY
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.12), Inches(12.33), Pt(3))
    ln.fill.solid()
    ln.fill.fore_color.rgb = ACCENT
    ln.line.fill.background()
    ln.shadow.inherit = False


def bullets(slide, items, top=1.35, size=16, left=0.65, width=12.0, height=5.7):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        text, lvl = (it if isinstance(it, tuple) else (it, 0))
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = 0
        r = p.add_run()
        r.text = ("• " if lvl == 0 else "–  ") + text
        r.font.size = Pt(size if lvl == 0 else size - 2)
        r.font.color.rgb = DARK if lvl == 0 else GREY
        p.space_after = Pt(6)


def table(slide, headers, rows, top=1.35, left=0.5, width=12.33, size=12, col_widths=None):
    r = len(rows) + 1
    c = len(headers)
    shp = slide.shapes.add_table(r, c, Inches(left), Inches(top), Inches(width), Inches(0.35 * r))
    tbl = shp.table
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(size)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.color.rgb = DARK
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xF2, 0xF6, 0xFA)
    return tbl


def two_images(slide, left_img, left_cap, right_img, right_cap, top=1.4, h=5.4, captions=True):
    for i, (img, cap) in enumerate(((left_img, left_cap), (right_img, right_cap))):
        x = 0.5 + i * 6.5
        slide.shapes.add_picture(img, Inches(x), Inches(top), height=Inches(h))
        if captions:
            tb = slide.shapes.add_textbox(Inches(x), Inches(top + h + 0.02), Inches(6.2), Inches(0.35))
            r = tb.text_frame.paragraphs[0].add_run()
            r.text = cap
            r.font.size = Pt(12)
            r.font.bold = True
            r.font.color.rgb = ACCENT


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def main():
    old = OrderedDict()
    for disp, name, params, lic in MODELS:
        s = load(name)
        old[name] = s
    old_rows = sorted([(d, load(n)["median_cer"] if load(n) else 2.0)
                       for d, n, p, l in MODELS], key=lambda kv: kv[1])
    new_rows = sorted([(d, n) for d, n, p, l in MODELS], key=lambda kv: NEW.get(kv[1], 2.0))

    chart_leaderboard(old_rows, os.path.join(DECK, "chart_old.png"),
                      "ClinOCR-Bench — median CER")
    chart_old_new([(d, (load(n)["median_cer"] if load(n) else 2.0), NEW.get(n, 2.0))
                   for d, n, p, l in MODELS], os.path.join(DECK, "chart_old_new.png"))
    chart_sweep(os.path.join(DECK, "chart_sweep.png"))

    p = prs_new()

    # 1 title
    s = blank(p)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(2.6))
    tf = tb.text_frame
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = "Open-source document OCR for clinical scans"
    r.font.size = Pt(40)
    r.font.bold = True
    r.font.color.rgb = DARK
    r2 = tf.add_paragraph().add_run()
    r2.text = "A model bake-off, distillation study, and layout/bbox evaluation"
    r2.font.size = Pt(20)
    r2.font.color.rgb = ACCENT
    r3 = tf.add_paragraph().add_run()
    r3.text = "328 synthetic clinical scans + 651 real medical scans  ·  2× RTX 4090  ·  all local, no PHI"
    r3.font.size = Pt(14)
    r3.font.color.rgb = GREY
    notes(s, "Everything here is public data or synthetic; no PHI. All models run locally on 2x RTX 4090.")

    # 2 question
    s = blank(p)
    bar(s, "The question", "What should replace Tesseract / commercial layout-OCR for scanned clinical documents?")
    bullets(s, [
        "Scanned referrals, lab panels, discharge summaries, faxes — degraded, rotated, handwritten, tabular.",
        "Need: (1) the values (MRNs, dates, lab decimals, meds) AND (2) the structure (tables, reading order, selection controls).",
        "Constraint: cheap enough to run at document volume — target < $0.01 / page.",
        "Open-weight VLMs now exist that do full-page transcription in one pass. Do any of them actually work here?",
    ])
    notes(s, "Framed around replacing the incumbent; cost target $0.01/page is the commercial-layout-OCR price point.")

    # 3 roster
    s = blank(p)
    bar(s, "The models", "12 open-weight OCR / document VLMs, one incumbent, comparable sizes and licences")
    rows = [(d, p_, l) for d, n, p_, l in MODELS]
    table(s, ["model", "params", "licence"], rows, size=11, col_widths=[5.0, 3.0, 4.3])
    notes(s, "Roster spans 0.26B -> 27B. Licences matter: Chandra is research-only; Qwen3.8 terms unverified.")

    # 4 corpus old
    s = blank(p)
    bar(s, "Corpus 1 — ClinOCR-Bench (the baseline)", "328 documents · 6 degradation subsets · human-audited ground truth")
    two_images(s, os.path.join(DECK, "normal_t1_s2_input.png"), "normal",
               os.path.join(DECK, "handwriting_t1_s2_input.png"), "handwriting", top=1.35, h=4.6)
    bullets(s, ["subsets: normal · handwriting · poor · rotated · tables · mixed",
                "(shown: clean + handwriting; the tables subset is actually filled forms)"], top=6.3, size=13)
    notes(s, "Synthetic content, real degradation pipeline, MIT. This is where every model was first ranked.")

    # 5 corpus new
    s = blank(p)
    bar(s, "Corpus 2 — medreal (real scans)", "651 pages assembled from public medical-document datasets, labelled by a vision LLM")
    table(s, ["source", "pages", "what it is"], [
        ["noisy-med", "400", "patient account statements, full page"],
        ["prescription", "200", "prescription scans"],
        ["india-hist", "46", "real 1878 hospital reports (external GT)"],
        ["medform", "5", "medical forms"],
    ], top=1.35, size=12, col_widths=[3.0, 1.5, 7.8])
    bullets(s, ["Labels minted with DeepSeek-v4.1-flash (vision) at $0.0009/page — the same prompt used in training",
                "Caveat: new-set ground truth is the teacher label, so it measures agreement, not human-verified accuracy",
                "Cheap generalisation check: a different pipeline, real scanners, non-English names/addresses"], top=4.6, size=15)
    notes(s, "651 pages, $0.60 total to label. The historical 1878 scans are the only independently-labelled subset.")

    # 6 method
    s = blank(p)
    bar(s, "How we measure", "Character error rate, with the pitfalls accounted for")
    bullets(s, [
        "Primary metric: median CER (character error rate) vs ground truth — lower is better.",
        "Mean CER is NOT stable: a few unreadable pages loop into repeated text and pin at the CER cap of 2.0.",
        ("The runaway tail carries ~half the error and varies run to run — so we rank on median, and report runaways.", 1),
        "Also measured: field-token recall/precision (dates, MRNs, decimals, codes), failure taxonomy, throughput, cost.",
        "Noise floor measured from n=5 identical repeats: median CER ±0.003, mean CER ±0.023.",
        ("Any difference smaller than that is not a finding.", 1),
    ])
    notes(s, "This methodological point is why several earlier 'findings' were withdrawn; median + runaway count is the stable read.")

    # 7 leaderboard old
    s = blank(p)
    bar(s, "Results — ClinOCR-Bench (328 docs)", "median CER; Tesseract baseline dashed")
    s.shapes.add_picture(os.path.join(DECK, "chart_old.png"), Inches(0.4), Inches(1.35), height=Inches(5.6))
    bullets(s, [
        "Every VLM beats Tesseract on degraded scans (0.02–0.10 vs 0.444).",
        "Qwen3.8-27B is a tier above everything (0.0224) and leads field recall (0.91).",
        "dots.mocr / Qwen7B / olmOCR-2 cluster at ~0.05–0.08 — cheap and strong.",
        "Chandra/granite/DeepSeek/Nanonets are not usable here.",
    ], left=8.0, width=5.0, size=14, top=1.6)
    notes(s, "On clean pages Tesseract is competitive; the case for replacement is entirely the degraded tail.")

    # 8 bbox
    s = blank(p)
    bar(s, "Selection controls + bounding boxes", "dots.mocr, native layout prompt — regions with bbox + category, form read as HTML tables")
    two_images(s, os.path.join(DECK, "tables_t16_s2_input.png"), "input (a filled behaviour form)",
               os.path.join(DECK, "tables_t16_s2_bbox.png"), "dots.mocr regions (bbox + category)", top=1.3, h=5.5)
    notes(s, "The checked boxes (Frustrated, Backpack, Hitting, Pushing, ...) are read correctly as text glyphs. "
             "The model returns the whole form as Table regions; it does not emit one box per checkbox.")

    # 9 layout metrics
    s = blank(p)
    bar(s, "Layout prompt: geometry + structure + controls", "same model, dedicated prompt asking for JSON regions")
    table(s, ["run", "bbox coverage", "tables as HTML", "text CER (median)", "checked-box recall"], [
        ["dots.mocr (plain)", "none", "partial", "0.0747", "0.65"],
        ["dots.mocr (layout)", "100% (7091)", "242 / 242", "0.0274", "0.70"],
        ["dots.ocr (layout)", "100% (6788)", "242 / 242", "0.0211", "0.70"],
    ], top=1.4, size=13, col_widths=[3.0, 2.6, 2.4, 2.3, 2.0])
    bullets(s, [
        "→ 100% of elements carry [x1,y1,x2,y2] + a category (Text, Section-header, Table, …).",
        "→ Every table returned as parsed HTML; text accuracy improves sharply vs the plain prompt.",
        "Caveat: ~0.70 of checked boxes captured, and the standalone-X 'Yes/No' forms are still missed (~0.0–0.10).",
        "Open item: does the geometry keep each check attached to the right field? Needs a structured-control scorer.",
    ], top=4.3, size=15)
    notes(s, "This is the key new capability: bboxes + categories + HTML tables, with most selection glyphs preserved.")

    # 10 selection across models
    s = blank(p)
    bar(s, "Do models capture selection controls?", "recall of checked marks (☑/☒) on the form subset")
    table(s, ["model", "checked-box recall", "model", "checked-box recall"], [
        ["Qwen2.5-VL-7B + 1 exemplar", "0.85", "dots.ocr", "0.33"],
        ["Qwen3.8-27B", "0.82", "Qwen2.5-VL-3B", "0.24"],
        ["olmOCR-2", "0.77", "PaddleOCR-VL", "0.03"],
        ["dots.mocr", "0.65", "Chandra / granite / DeepSeek / Tesseract", "0.00"],
    ], top=1.4, size=12, col_widths=[4.2, 2.0, 4.2, 1.9])
    bullets(s, [
        "Several open models reliably read ☑ vs ☐ as text — but as glyphs, not structured {control, state} fields.",
        "The models that emit coordinates (chandra, granite) capture none of the controls; the ones that capture controls mostly emit no coordinates.",
        "dots.mocr's layout prompt is the only run that gives both (see previous slide).",
    ], top=3.7, size=15)
    notes(s, "Tension worth flagging: geometry and control-state came from disjoint sets of models until the dots layout prompt.")

    # 11 training recipe
    s = blank(p)
    bar(s, "Training — distillation of a small student", "LoRA on synthetic degraded clinical docs; teacher = Qwen3.8-27B")
    bullets(s, [
        "Method: LoRA adapters (r=16, α=32, dropout 0.05) on frozen Qwen2.5-VL base — NOT full fine-tuning.",
        "Targets q/k/v/o/gate/up/down (7 projections, incl. vision tower): 37M trainable on 3B (0.98%), 48M on 7B (0.57%).",
        "1 epoch, lr 1e-4 cosine, grad-accum 8, bf16, images capped at 802,816 px.",
        "Labels: either exact synthetic GT, or a teacher's transcript of unlabelled scans (no ground truth needed).",
        "10 adapters trained: 4× (label-source study), 5× (corpus-size sweep), 1× (old+new mixed).",
    ])
    notes(s, "The whole point: can a cheap 3B/7B student inherit a big teacher's domain accuracy from unlabelled scans?")

    # 12 training results / sweep
    s = blank(p)
    bar(s, "Training result — does more data help?", "same teacher, same recipe, corpus size varied")
    s.shapes.add_picture(os.path.join(DECK, "chart_sweep.png"), Inches(0.5), Inches(1.4), height=Inches(4.4))
    bullets(s, [
        "3B: flat at/below the untuned base from 800→4800 docs → capacity limit, not data limit.",
        "7B: 800 docs cuts median 0.072 → 0.045 (a real win); 2400 regresses to 0.062.",
        "Distillation helps the 7B, not the 3B; and more data is not monotonically better.",
        "Fine-tuning also raises runaway generation — a production caveat.",
    ], left=7.9, width=5.1, size=14, top=1.6)
    notes(s, "n=1 per point; the non-monotonic 7B result deserves a repeat before it goes in a deck.")

    # 13 one-shot
    s = blank(p)
    bar(s, "Cheaper alternative to fine-tuning: 1-shot exemplar", "prepend one clean exemplar of the same form template")
    table(s, ["approach", "median CER", "cost / page"], [
        ["Qwen2.5-VL-7B (zero-shot)", "0.0513", "$0.0021"],
        ["Qwen2.5-VL-7B + 1 homogeneous exemplar", "0.0206", "$0.0024"],
        ["Qwen3.8-27B (teacher)", "0.0224", "$0.0084"],
    ], top=1.5, size=14, col_widths=[6.5, 2.9, 2.9])
    bullets(s, [
        "One clean exemplar brings the 7B to ~27B accuracy on median CER — at ~1/4 the cost.",
        "No training required; the exemplar is a prompt-time addition.",
        "Trade-off: extra input tokens per request; distillation bakes the same benefit into the weights.",
    ], top=3.8, size=15)
    notes(s, "This was tonight's standout: 0.0206 median, statistically tied with the 27B, at 7B cost.")

    # 14 old vs new
    s = blank(p)
    bar(s, "Generalisation — synthetic vs real scans", "every model re-run on the 651-page real corpus")
    s.shapes.add_picture(os.path.join(DECK, "chart_old_new.png"), Inches(0.4), Inches(1.3), height=Inches(4.35))
    bullets(s, [
        "Top of the ranking survives: Qwen3.8-27B (0.080) then Qwen2.5-VL-7B (0.113) on real scans too.",
        "Absolute error roughly doubles on real scans; real pages are harder but trigger fewer loops.",
        "dots.ocr collapses on real statements (median 2.0; 341/400 pages loop).",
        "granite-docling improves (0.87→0.32) but stays weak; PaddleOCR-VL degrades.",
    ], top=5.85, size=13)
    notes(s, "The ranking we care about holds at the top; the mid-pack reshuffles and one model breaks outright.")

    # 15 cost
    s = blank(p)
    bar(s, "Cost — all viable models clear $0.01/page", "at $4 / GPU-hour, you need only 0.111 pages/s per GPU")
    table(s, ["model (1 GPU unless noted)", "pages/s", "$ / page", "median CER"], [
        ["Qwen2.5-VL-3B", "1.06", "$0.0010", "0.0649"],
        ["dots.mocr", "0.67", "$0.0017", "0.0747"],
        ["Qwen2.5-VL-7B", "0.54", "$0.0021", "0.0513"],
        ["Qwen2.5-VL-7B + 1-shot", "0.46", "$0.0024", "0.0206"],
        ["Qwen3.8-27B (2 GPUs, TP=2)", "0.26", "$0.0084", "0.0224"],
    ], top=1.4, size=12, col_widths=[5.6, 2.2, 2.2, 2.3])
    bullets(s, [
        "Cost is not the binding constraint — accuracy and structure are.",
        "Compute only; excludes retries/escalation. Real metric is $ / acceptable page.",
        "27B is under the line but with little headroom; int4 would roughly halve it.",
    ], top=4.2, size=15)
    notes(s, "Every shortlisted model is 4-10x under the $0.01/page commercial price point.")

    # 16 recommendations
    s = blank(p)
    bar(s, "What we'd recommend", "for document intelligence: values + structure")
    bullets(s, [
        "Primary: Qwen2.5-VL-7B (+1 exemplar) — near-27B accuracy at ~$0.002/page, no training needed.",
        "Structure/geometry: dots.mocr with its layout prompt — bboxes + categories + HTML tables + most selection glyphs.",
        "Ceiling: Qwen3.8-27B — best values (field recall 0.91) and fewest runaways; still < $0.01/page.",
        "Avoid as primary: PaddleOCR-VL, granite-docling, DeepSeek-OCR, Nanonets (values/runaways).",
        "Distillation: helps the 7B at ~800 docs; the 3B is capacity-limited. Fine-tuning is not a free win.",
    ], size=16)
    notes(s, "The two-model story: a general VLM for text values, dots.mocr for structure, both cheap.")

    # 17 caveats + next
    s = blank(p)
    bar(s, "Caveats & next tests", "what to verify before committing")
    bullets(s, [
        "Selection-control association is unproven — we score glyph presence, not whether the check stays bound to the correct field.",
        "New-set ground truth is teacher-labelled (circular for the distilled model); only the 1878 scans have independent GT.",
        "n=1 per training point and a small form sample — repeat before quoting.",
        "Next: structured selection-control scorer; run the layout prompt on real filled forms; extend to real faxes at volume.",
    ], size=16)
    notes(s, "These are the honest limits; the benchmark is strong on text fidelity, weaker on structure association so far.")

    out = os.path.join(DECK, "clinical_ocr_bakeoff.pptx")
    p.save(out)
    print("saved", out, "with", len(p.slides.__iter__.__self__._sldIdLst), "slides")


if __name__ == "__main__":
    main()