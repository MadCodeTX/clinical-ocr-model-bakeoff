#!/usr/bin/env python3
"""PaddleOCR-VL FULL PIPELINE eval: PP-DocLayoutV2 layout detection (+ optional
orientation classification / unwarping) -> region crops -> PaddleOCR-VL via a
running vLLM server -> assembled markdown.

Runs in paddle-venv. Writes predictions.jsonl; scoring is done separately by
score_preds.py (main venv) so this env needs no rapidfuzz.

Usage:
  python3 eval_paddle_pipeline.py --out results/paddleocr-vl-pipeline \
      --server http://localhost:8000/v1 --limit 0
"""
import argparse
import json
import os
import shutil
import tempfile
import time
import traceback

from paddleocr import PaddleOCRVL

ROOT = os.path.dirname(os.path.abspath(__file__))


def build_pipeline(server, use_orient, use_unwarp):
    kwargs = {
        "vl_rec_backend": "vllm-server",
        "vl_rec_server_url": server,
        "device": "cpu",
    }
    if use_orient:
        kwargs["use_doc_orientation_classify"] = True
    if use_unwarp:
        kwargs["use_doc_unwarping"] = True
    try:
        return PaddleOCRVL(**kwargs)
    except TypeError as e:
        print("warn: option not supported, retrying minimal kwargs:", e)
        return PaddleOCRVL(vl_rec_backend="vllm-server",
                           vl_rec_server_url=server, device="cpu")


def prep_image(src, tmpdir, i, max_side=1600):
    """Normalise to RGB PNG with a bounded long side. The paddle 'cv' worker
    throws std::exception on some JPEGs / odd geometries; feeding it a clean
    RGB PNG (and capping size) removes that failure mode."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    path = os.path.join(tmpdir, f"prep_{i}.png")
    im.save(path)
    return path


def extract_text(res, tmpdir, i):
    """Try several APIs to get markdown/text out of a PaddleOCR result."""
    for path in (os.path.join(tmpdir, f"o{i}.md"), tmpdir):
        try:
            res.save_to_markdown(save_path=path)
            p = path if path.endswith(".md") else os.path.join(path, f"o{i}.md")
            if os.path.exists(p):
                return open(p).read()
        except Exception:
            pass
    for attr in ("markdown", "text"):
        v = getattr(res, attr, None)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for k in ("markdown_text", "markdown", "text"):
                if k in v:
                    return str(v[k])
    try:  # last resort: JSON blob
        return json.dumps(res.json) if hasattr(res, "json") else str(res)
    except Exception:
        return str(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "clinocr"))
    ap.add_argument("--server", default="http://localhost:8000/v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-orient", action="store_true")
    ap.add_argument("--no-unwarp", action="store_true")
    ap.add_argument("--no-preprocess", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.data, "eval.jsonl")) as f:
        items = [json.loads(l) for l in f]
    if args.limit:
        items = items[:args.limit]

    print(f"loading PaddleOCRVL pipeline (server={args.server}) ...", flush=True)
    pipeline = build_pipeline(args.server, not args.no_orient, not args.no_unwarp)
    print("pipeline ready", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tmpdir = tempfile.mkdtemp()
    t0 = time.time()
    with open(os.path.join(args.out, "predictions.jsonl"), "w") as f:
        for i, it in enumerate(items):
            img = os.path.join(ROOT, it["image"])
            t = time.time()
            text, err = "", None
            try:
                src = img if args.no_preprocess else prep_image(img, tmpdir, i)
                out = pipeline.predict(src)
                parts = []
                for j, res in enumerate(out):
                    parts.append(extract_text(res, tmpdir, j))
                text = "\n".join(parts)
            except Exception:
                err = traceback.format_exc(limit=2)
            f.write(json.dumps({
                "doc_id": it["doc_id"], "subset": it["subset"],
                "latency_s": round(time.time() - t, 2),
                "prediction": text, "error": err}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(items)}  {el:.0f}s  "
                      f"eta {el/(i+1)*(len(items)-i-1):.0f}s", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"done in {time.time()-t0:.0f}s -> {args.out}/predictions.jsonl")


if __name__ == "__main__":
    main()
