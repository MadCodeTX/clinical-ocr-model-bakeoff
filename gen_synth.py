#!/usr/bin/env python3
"""Generate synthetic degraded clinical documents with EXACT ground truth.

Renders clinical-style documents (referral fax, lab report, discharge summary,
letterhead report) with randomized content, then applies realistic scan
degradations: rotation, perspective warp, blur, sensor noise, low-DPI resample,
JPEG artifacts, speckles, fold lines, vignette. Handwriting-font variants
emulate hand-annotated forms.

Ground truth is the exact rendered text => free, perfect labels. This corpus
trains the distillation student; evaluation stays on the untouched
ClinOCR-Bench test split (built by a different pipeline => no leakage).

Usage: python3 gen_synth.py --n 800 --out data/synth
"""
import argparse
import json
import math
import os
import random
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_CANDIDATES = {
    "sans": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    "sans_bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
    "serif": ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
    "mono": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}
HANDWRITING_URL = ("https://github.com/google/fonts/raw/main/ofl/patrickhand/"
                   "PatrickHand-Regular.ttf")

FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
         "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
         "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
         "Nancy", "Matthew", "Lisa", "Anthony", "Betty", "Mark", "Margaret"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark"]
FACILITIES = ["Springfield General Hospital", "Lakewood Medical Center",
              "Riverside Regional Hospital", "St. Mary's Health Center",
              "Grandview Medical Center", "Mercy General Hospital",
              "Cedar Ridge Clinic", "Northshore Family Practice",
              "Midwest Heart & Vascular Institute", "Prairie Pathology Lab"]
DEPTS = ["Cardiology", "Internal Medicine", "Gastroenterology", "Oncology",
         "Radiology", "Pathology", "Family Medicine", "Pulmonology",
         "Endocrinology", "Nephrology", "Orthopedics", "Neurology"]
STREETS = ["Health Way", "Oak Avenue", "Evergreen Terrace", "Cardiovascular Circle",
           "Medical Plaza", "Lakeview Drive", "Main Street", "Park Boulevard"]
CITIES = [("Springfield", "IL", "62704"), ("Lakewood", "CO", "80215"),
          ("Anytown", "USA", "12345"), ("Lincolnwood", "IL", "60712"),
          ("Fairview", "TX", "75069"), ("Riverton", "CA", "95605")]
DRUGS = ["Lisinopril 10 mg PO daily", "Metformin 500 mg PO BID",
         "Atorvastatin 40 mg PO nightly", "Levothyroxine 75 mcg PO daily",
         "Amlodipine 5 mg PO daily", "Metoprolol succinate 25 mg PO daily",
         "Omeprazole 20 mg PO daily", "Gabapentin 300 mg PO TID",
         "Hydrochlorothiazide 25 mg PO daily", "Warfarin 5 mg PO daily"]
DX = ["essential hypertension", "type 2 diabetes mellitus",
      "hyperlipidemia", "chronic kidney disease stage 3",
      "atrial fibrillation", "congestive heart failure",
      "chronic obstructive pulmonary disease", "hypothyroidism",
      "gastroesophageal reflux disease", "osteoarthritis"]
LABS = [("Sodium", "136-145", "mmol/L", (128, 148)),
        ("Potassium", "3.5-5.1", "mmol/L", (3.0, 5.8)),
        ("Chloride", "98-107", "mmol/L", (94, 112)),
        ("CO2", "22-29", "mmol/L", (18, 32)),
        ("BUN", "6-20", "mg/dL", (5, 42)),
        ("Creatinine", "0.6-1.3", "mg/dL", (0.5, 2.4)),
        ("Glucose", "70-99", "mg/dL", (62, 210)),
        ("Calcium", "8.6-10.3", "mg/dL", (7.9, 11.0)),
        ("Albumin", "3.5-5.0", "g/dL", (2.8, 5.2)),
        ("Hemoglobin", "12.0-16.0", "g/dL", (9.1, 17.2)),
        ("WBC", "4.0-11.0", "K/uL", (3.1, 14.8)),
        ("Platelets", "150-450", "K/uL", (98, 520))]
SYMPTOMS = ["chest pain and shortness of breath for two days",
            "progressive fatigue and unintentional weight loss",
            "persistent cough with intermittent fever",
            "recurrent abdominal pain and bloating after meals",
            "dizziness and near-syncope on standing",
            "non-painful pigmented lesion on the left forearm"]
PLANS = ["Continue current medications and recheck labs in 3 months.",
         "Refer to cardiology for stress testing and echocardiogram.",
         "Start physical therapy twice weekly for 6 weeks.",
         "Repeat imaging in 4 weeks to assess interval change.",
         "Patient educated on diet, exercise, and medication adherence."]
PROC = ["Colonoscopy", "Echocardiogram", "CT Abdomen/Pelvis with contrast",
        "Upper GI Endoscopy", "Stress Test", "MRI Brain without contrast"]


def load_fonts(out_dir):
    fonts = {}
    for kind, paths in FONT_CANDIDATES.items():
        for p in paths:
            if os.path.exists(p):
                fonts[kind] = p
                break
    # handwriting font (best effort)
    hw_path = os.path.join(out_dir, "PatrickHand-Regular.ttf")
    if not os.path.exists(hw_path):
        try:
            urllib.request.urlretrieve(HANDWRITING_URL, hw_path)
        except Exception:
            hw_path = None
    if hw_path and os.path.exists(hw_path):
        fonts["handwriting"] = hw_path
    fonts.setdefault("handwriting", fonts.get("serif", fonts.get("sans")))
    return fonts


class Doc:
    """Renders a page while recording exact text lines."""

    def __init__(self, width, height, fonts, hw=False):
        self.img = Image.new("RGB", (width, height), "white")
        self.d = ImageDraw.Draw(self.img)
        self.fonts = fonts
        self.lines = []
        self.y = 40
        self.margin = 60
        self.hw = hw

    def font(self, kind, size):
        if self.hw and kind in ("sans", "serif"):
            kind = "handwriting"
        path = self.fonts.get(kind) or self.fonts["sans"]
        return ImageFont.truetype(path, size)

    def text(self, s, kind="sans", size=20, gap=6, indent=0, align="left"):
        f = self.font(kind, size)
        bbox = self.d.textbbox((0, 0), s, font=f)
        w = bbox[2] - bbox[0]
        x = self.margin + indent
        if align == "center":
            x = (self.img.width - w) // 2
        elif align == "right":
            x = self.img.width - self.margin - w
        self.d.text((x, self.y), s, fill="black", font=f)
        self.lines.append(s)
        self.y += (bbox[3] - bbox[1]) + gap
        return self

    def rule(self, gap=10):
        self.d.line([(self.margin, self.y), (self.img.width - self.margin, self.y)],
                    fill="black", width=2)
        self.y += gap

    def wrap(self, s, kind="sans", size=19, width=None, gap=5):
        f = self.font(kind, size)
        maxw = width or (self.img.width - 2 * self.margin)
        words, line = s.split(), ""
        for w in words:
            t = (line + " " + w).strip()
            if self.d.textlength(t, font=f) > maxw and line:
                self.text(line, kind, size, gap)
                line = w
            else:
                line = t
        if line:
            self.text(line, kind, size, gap)
        return self

    def paragraph(self, n=3, kind="sans", size=19):
        for _ in range(n):
            self.wrap(" ".join(random.sample(SYMPTOMS + PLANS, 1)) + " " +
                      "The patient was evaluated and the findings were discussed. "
                      "Assessment and plan documented below.", kind, size)
            self.y += 6
        return self

    def table(self, headers, rows, size=17, col_w=None):
        n = len(headers)
        avail = self.img.width - 2 * self.margin
        col_w = col_w or [avail // n] * n
        f = self.font("sans", size)
        self.d.line([(self.margin, self.y), (self.img.width - self.margin, self.y)],
                    fill="black", width=2)
        self.y += 6
        x = self.margin
        for h, w in zip(headers, col_w):
            self.d.text((x + 6, self.y), h, fill="black", font=f)
            x += w
        self.lines.append(" | ".join(headers))
        self.y += size + 8
        self.d.line([(self.margin, self.y), (self.img.width - self.margin, self.y)],
                    fill="black", width=1)
        self.y += 6
        for r in rows:
            x = self.margin
            for cell, w in zip(r, col_w):
                self.d.text((x + 6, self.y), str(cell), fill="black", font=f)
                x += w
            self.lines.append(" | ".join(str(c) for c in r))
            self.y += size + 6
            self.d.line([(self.margin, self.y),
                         (self.img.width - self.margin, self.y)], fill="gray", width=1)
            self.y += 4
        self.y += 8
        return self

    def image(self):
        return self.img


def person():
    return f"{random.choice(LAST)}, {random.choice(FIRST)}"


def doc_referral(d):
    fac = random.choice(FACILITIES)
    street = f"{random.randint(100, 999)} {random.choice(STREETS)}"
    city, st, zc = random.choice(CITIES)
    d.text(fac, "sans_bold", 30, 4, align="center")
    d.text(f"{street}, {city}, {st} {zc}", "sans", 16, 2, align="center")
    d.text(f"Phone: ({random.randint(200,989)}) 555-{random.randint(1000,9999)}   "
           f"Fax: ({random.randint(200,989)}) 555-{random.randint(1000,9999)}",
           "sans", 16, 10, align="center")
    d.rule()
    d.text("FAX", "sans_bold", 34, 8, align="center")
    d.text(f"To: Dr. {random.choice(FIRST)} {random.choice(LAST)}, "
           f"{random.choice(DEPTS)}", "sans", 20, 6)
    d.text(f"Facility: {random.choice(FACILITIES)}", "sans", 20, 6)
    d.text(f"Fax: (555) {random.randint(100,999)}-{random.randint(1000,9999)}", "sans", 20, 6)
    d.text(f"Date: {random.randint(1,12)}/{random.randint(1,28)}/2025", "sans", 20, 6)
    d.text(f"Re: Consult for Patient: {person()}", "sans", 20, 6)
    d.text(f"MRN: {random.randint(100000,9999999)}", "sans", 20, 6)
    d.text(f"DOB: {random.randint(1,12)}/{random.randint(1,28)}/19{random.randint(40,99)}",
           "sans", 20, 6)
    d.rule()
    d.text("Dear Colleague,", "sans", 20, 8)
    d.wrap(f"Thank you for seeing this {random.randint(30,88)}-year-old patient with a "
           f"history of {random.choice(DX)} and {random.choice(DX)}. The patient "
           f"presents with {random.choice(SYMPTOMS)}.", "sans", 19)
    d.y += 8
    d.wrap("Recent laboratory evaluation and imaging are enclosed. Please evaluate "
           "and advise on further management.", "sans", 19)
    d.y += 20
    d.text("Sincerely,", "sans", 19, 30)
    d.text(f"Dr. {random.choice(FIRST)} {random.choice(LAST)}, MD", "sans", 19, 4)
    d.text(f"{random.choice(DEPTS)}", "sans", 19, 4)
    d.rule()
    d.text("CONFIDENTIALITY NOTICE: This facsimile is intended only for the use of "
           "the individual named above.", "sans", 13, 88, align="center")


def doc_labs(d):
    d.text(random.choice(FACILITIES), "sans_bold", 28, 4, align="center")
    d.text(random.choice(DEPTS) + " Laboratory", "sans", 18, 2, align="center")
    d.text("COMPREHENSIVE METABOLIC PANEL", "sans_bold", 24, 12, align="center")
    d.text(f"Patient: {person()}", "sans", 18, 4)
    d.text(f"MRN: {random.randint(100000,9999999)}   "
           f"DOB: {random.randint(1,12)}/{random.randint(1,28)}/19{random.randint(40,99)}",
           "sans", 18, 4)
    d.text(f"Date of Service: {random.randint(1,12)}/{random.randint(1,28)}/2025   "
           f"Accession #: L{random.randint(10000,99999)}", "sans", 18, 12)
    rows = []
    for name, ref, unit, (lo, hi) in random.sample(LABS, random.randint(7, 10)):
        val = round(random.uniform(lo * 0.85, hi * 1.15), 2 if "Creatinine" == name else 1)
        flag = "H" if val > hi else ("L" if val < lo else "")
        rows.append([name, val, flag, ref, unit])
    rows.append(["", "", "", "", ""])
    d.table(["Test", "Result", "Flag", "Reference", "Units"], rows,
            col_w=[190, 110, 80, 150, 130])
    d.text(f"Comments: {random.choice(['Results verified by repeat analysis.',
                                      'Specimen mildly hemolyzed; interpret with caution.',
                                      'Critical value called to ordering provider.'])}",
           "sans", 16, 6)
    d.text(f"Ordering Physician: Dr. {random.choice(FIRST)} {random.choice(LAST)}",
           "sans", 18, 6)
    d.text("Electronically signed by Laboratory Director", "sans", 16, 40)


def doc_discharge(d):
    d.text(random.choice(FACILITIES), "sans_bold", 28, 4, align="center")
    d.text("DISCHARGE SUMMARY", "sans_bold", 26, 12, align="center")
    d.text(f"Patient Name: {person()}", "sans", 18, 4)
    d.text(f"MRN: {random.randint(100000,9999999)}    "
           f"Admission Date: {random.randint(1,12)}/{random.randint(1,28)}/2025", "sans", 18, 4)
    d.text(f"Discharge Date: {random.randint(1,12)}/{random.randint(1,28)}/2025    "
           f"Attending: Dr. {random.choice(FIRST)} {random.choice(LAST)}, MD", "sans", 18, 12)
    d.rule()
    for section in ["Chief Complaint", "Hospital Course", "Assessment",
                    "Discharge Medications", "Follow-up Instructions"]:
        d.text(section, "sans_bold", 21, 6)
        if section == "Discharge Medications":
            for _ in range(random.randint(3, 6)):
                d.text(f"{random.randint(1,3)}. {random.choice(DRUGS)}", "sans", 18, 3, indent=20)
        elif section == "Assessment":
            for _ in range(random.randint(2, 4)):
                d.text(f"- {random.choice(DX).capitalize()}", "sans", 18, 3, indent=20)
        else:
            d.paragraph(random.randint(1, 2), size=18)
        d.y += 4
    d.rule()
    d.text(f"Procedure performed: {random.choice(PROC)}", "sans", 18, 4)
    d.text(f"Condition at discharge: {random.choice(['Stable', 'Improved', 'Fair'])}",
           "sans", 18, 4)


def doc_path(d):
    d.text(random.choice(FACILITIES) + " - Pathology", "sans_bold", 28, 8, align="center")
    d.text("SURGICAL PATHOLOGY REPORT", "sans_bold", 24, 12, align="center")
    d.text(f"Patient Name: {person()}", "sans", 18, 4)
    d.text(f"Date of Birth: {random.randint(1,12)}/{random.randint(1,28)}/19{random.randint(40,99)}"
           f"   MRN: {random.randint(100000,9999999)}", "sans", 18, 4)
    d.text(f"Accession #: S{random.randint(10000,99999)}   "
           f"Date of Report: {random.randint(1,12)}/{random.randint(1,28)}/2025", "sans", 18, 12)
    d.rule()
    d.text("Clinical Information", "sans_bold", 20, 6)
    d.wrap(f"{random.randint(30,88)}-year-old patient with {random.choice(DX)} "
           f"presenting with {random.choice(SYMPTOMS)}. Specimen submitted for "
           f"histologic evaluation.", "sans", 18)
    d.y += 8
    d.text("Gross Description", "sans_bold", 20, 6)
    d.paragraph(1, size=18)
    d.y += 8
    d.text("Microscopic Description", "sans_bold", 20, 6)
    d.paragraph(1, size=18)
    d.y += 8
    d.text("Final Diagnosis", "sans_bold", 20, 6)
    d.wrap("Benign tissue with no evidence of malignancy. Margins are negative. "
           "Immunohistochemical stains are consistent with reactive change.", "sans", 18)
    d.y += 16
    d.text(f"Pathologist: Dr. {random.choice(FIRST)} {random.choice(LAST)}, MD",
           "sans", 18, 4)
    d.text(f"Reported: {random.randint(1,12)}/{random.randint(1,28)}/2025",
           "sans", 18, 4)


DOC_TYPES = [doc_referral, doc_labs, doc_discharge, doc_path]


# ------------------------- degradations -------------------------
def deg_rotate(im, rng):
    a = rng.uniform(-12, 12)
    return im.rotate(a, resample=Image.BICUBIC, expand=False, fillcolor="white")


def deg_perspective(im, rng):
    w, h = im.size
    j = lambda: rng.uniform(-0.02, 0.02) * w  # noqa: E731
    coeffs = (1 + j() / w, j() / w, j(), j() / h, 1 + j() / h, j(), 0, 0)
    return im.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def deg_blur(im, rng):
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 2.2)))


def deg_noise(im, rng):
    a = np.asarray(im).astype(np.float32)
    a += np.random.normal(0, rng.uniform(3, 22), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def deg_resample(im, rng):
    w, h = im.size
    s = rng.uniform(0.35, 0.75)
    small = im.resize((max(64, int(w * s)), max(64, int(h * s))), Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC)


def deg_contrast(im, rng):
    a = np.asarray(im).astype(np.float32) * rng.uniform(0.75, 1.15)
    a += rng.uniform(-25, 25)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def deg_speckle(im, rng):
    a = np.asarray(im).copy()
    h, w, _ = a.shape
    n = int(h * w * rng.uniform(0.00002, 0.00035))
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    a[ys, xs] = np.random.randint(0, 90, (n, 1))
    return Image.fromarray(a)


def deg_fold(im, rng):
    a = np.asarray(im).astype(np.float32)
    h, w, _ = a.shape
    if rng.random() < 0.5:
        y = int(h * rng.uniform(0.25, 0.75))
        a[max(0, y - 2):y + 2, :] *= rng.uniform(0.55, 0.8)
    else:
        x = int(w * rng.uniform(0.25, 0.75))
        a[:, max(0, x - 2):x + 2] *= rng.uniform(0.55, 0.8)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def deg_vignette(im, rng):
    a = np.asarray(im).astype(np.float32)
    h, w, _ = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    mask = np.clip(1 - rng.uniform(0.05, 0.35) * r, 0.35, 1.0)[..., None]
    return Image.fromarray(np.clip(a * mask, 0, 255).astype(np.uint8))


DEGS = [deg_rotate, deg_perspective, deg_blur, deg_noise, deg_resample,
        deg_contrast, deg_speckle, deg_fold, deg_vignette]


def degrade(im, rng, strength):
    applied = []
    for fn in DEGS:
        if rng.random() < strength:
            im = fn(im, rng)
            applied.append(fn.__name__.replace("deg_", ""))
    return im, applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", type=int, default=0, help="first index (for sharding)")
    ap.add_argument("--strength", type=float, default=0.55)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    fonts = load_fonts(args.out)
    if "handwriting" in fonts and HANDWRITING_URL.split("/")[-1].startswith("Patrick"):
        print("handwriting font:", fonts["handwriting"])

    labels_path = os.path.join(args.out, "labels.jsonl")
    if args.start == 0:
        open(labels_path, "w").close()
    fout = open(labels_path, "a")
    for i in range(args.start, args.start + args.n):
        hw = rng.random() < 0.3
        d = Doc(1275, 1650, fonts, hw=hw)
        rng.choice(DOC_TYPES)(d)
        img = d.image()
        img, applied = degrade(img, rng, args.strength)
        q = rng.randint(25, 80)
        name = f"synth_{i:05d}.jpg"
        path = os.path.join(args.out, "images", name)
        img.save(path, "JPEG", quality=q)
        fout.write(json.dumps({
            "id": f"synth_{i:05d}",
            "image": os.path.relpath(path, os.path.dirname(args.out.rstrip("/"))),
            "text": "\n".join(d.lines),
            "handwriting": hw,
            "degradations": applied,
            "jpeg_quality": q,
        }) + "\n")
        fout.flush()
        if (i - args.start + 1) % 50 == 0:
            print(f"  {i - args.start + 1}/{args.n}")
    fout.close()
    print(f"wrote {args.n} docs -> {args.out}")


if __name__ == "__main__":
    main()
