#!/usr/bin/env python3
"""HF-transformers eval for the student (base and LoRA) on ClinOCR-Bench test.

Same prompt/method as the vLLM runs, so CER numbers are comparable. Writes
predictions.jsonl for score_preds.py.

Usage: python3 eval_hf.py --out results/qwen25vl3b-base \
    [--adapter checkpoints/qwen25vl3b-lora] [--limit 0]
"""
import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    _KW = {}
except ImportError:  # transformers >= 5.x
    from transformers import AutoModelForImageTextToText as VLModel
    _KW = {"dtype": torch.bfloat16}

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = "Extract the text content from this image."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "clinocr", "eval.jsonl"))
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing predictions.jsonl, skipping finished docs")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.data)]
    if args.limit:
        items = items[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    # A run killed by its time budget leaves a valid partial file (one flushed
    # JSON line per doc); --resume picks up the docs it never reached.
    preds_path = os.path.join(args.out, "predictions.jsonl")
    done = set()
    if args.resume and os.path.exists(preds_path):
        with open(preds_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["doc_id"])
                except (json.JSONDecodeError, KeyError):
                    pass  # drop a torn final line rather than trusting it
        items = [it for it in items if it["doc_id"] not in done]
        print(f"resume: {len(done)} done, {len(items)} remaining", flush=True)
        if not items:
            print("nothing to do")
            return

    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = 4 * 28 * 28
    model = VLModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0", **_KW)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("loaded adapter", args.adapter)
    model.eval()

    t0 = time.time()
    with open(preds_path, "a" if args.resume else "w") as f:
        for i, it in enumerate(items):
            img = Image.open(os.path.join(ROOT, it["image"])).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": args.prompt}]}]
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")
            t = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            pred = processor.decode(gen, skip_special_tokens=True)
            f.write(json.dumps({"doc_id": it["doc_id"], "subset": it["subset"],
                                "latency_s": round(time.time() - t, 2),
                                "prediction": pred, "error": None}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(items)}  {el:.0f}s  "
                      f"eta {el/(i+1)*(len(items)-i-1):.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
