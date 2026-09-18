#!/usr/bin/env python3
"""Extended student trainer for the "can training help?" experiments.

Superset of train_student.py, adding the levers the failure analysis points at:

  * resolution          --max-pixels (train at the resolution you will serve at)
  * adaptation strength  LoRA rank/alpha, --target-scope {both,vision,lm}, or --full-ft
  * label quality        --min-chars/--max-chars/--drop-repetition filtering
  * optimisation         --epochs, --lr, --warmup, --optim {adamw,adafactor},
                         multi-seed (--seed), held-out --val-file with
                         best-checkpoint selection (--save-best)
  * checkpointing        --save-steps / --epochs so intermediate checkpoints can
                         be evaluated on the real benchmark

Usage:
  CUDA_VISIBLE_DEVICES=1 train-venv/bin/python3 train_student2.py \
     --model Qwen/Qwen2.5-VL-7B-Instruct --data data/synth_large/teacher_labels_q38.jsonl \
     --val-file data/val/synth_val.jsonl --out checkpoints/exp1 \
     --epochs 3 --lr 5e-5 --rank 64 --seed 1 --max-pixels 802816 --save-best
"""
import argparse
import inspect
import json
import os
import random

import torch
from PIL import Image

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

from transformers import AutoProcessor, Trainer, TrainingArguments

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    _LOAD_KW = {}
except ImportError:  # transformers >= 5.x
    from transformers import AutoModelForImageTextToText as VLModel
    _LOAD_KW = {"dtype": torch.bfloat16}

from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT = "Extract the text content from this image."
IMG_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


def looks_repetitive(text, thresh=0.4):
    toks = text.split()
    if len(toks) < 20:
        return False
    grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 4)]
    if not grams:
        return False
    from collections import Counter
    return Counter(grams).most_common(1)[0][1] / len(grams) > thresh


def load_rows(path, mn, mx, drop_rep):
    rows = [json.loads(l) for l in open(path)]
    kept, dropped = [], {"short": 0, "long": 0, "repetition": 0}
    for r in rows:
        t = (r.get("text") or "").strip()
        if len(t) < mn:
            dropped["short"] += 1
            continue
        if mx and len(t) > mx:
            dropped["long"] += 1
            continue
        if drop_rep and looks_repetitive(t):
            dropped["repetition"] += 1
            continue
        r["text"] = t
        kept.append(r)
    return kept, dropped


class SynthDocs(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = r["image"]
        for cand in (img, os.path.join(ROOT, img), os.path.join(ROOT, "data", img)):
            if os.path.exists(cand):
                return {"image": Image.open(cand).convert("RGB"), "text": r["text"]}
        raise FileNotFoundError(r["image"])


class VLMCollator:
    def __init__(self, processor):
        self.processor = processor
        self.im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

    def __call__(self, batch):
        texts, images = [], []
        for b in batch:
            msgs = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": [{"type": "text", "text": b["text"]}]},
            ]
            texts.append(self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))
            images.append(b["image"])
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            pos = (inputs["input_ids"][i] == self.im_start).nonzero()
            if len(pos):
                labels[i, :pos[-1].item() + 3] = -100
        inputs["labels"] = labels
        return inputs


def pick_targets(model, scope):
    names = [n for n, _ in model.named_modules() if n.endswith(IMG_SUFFIXES)]
    if scope == "vision":
        names = [n for n in names if "visual" in n]
    elif scope == "lm":
        names = [n for n in names if "visual" not in n]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "synth", "labels.jsonl"))
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target-scope", choices=["both", "vision", "lm"], default="both")
    ap.add_argument("--full-ft", action="store_true")
    ap.add_argument("--optim", default="adamw_torch", choices=["adamw_torch", "adafactor", "adamw_8bit"])
    ap.add_argument("--lr-scheduler", default="cosine")
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--drop-repetition", action="store_true")
    ap.add_argument("--save-steps", type=int, default=0, help="0 = epoch boundary")
    ap.add_argument("--save-total-limit", type=int, default=6)
    ap.add_argument("--save-best", action="store_true", help="requires --val-file")
    ap.add_argument("--eval-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows, dropped = load_rows(args.data, args.min_chars, args.max_chars, args.drop_repetition)
    if args.limit:
        rows = rows[:args.limit]
    val_rows = []
    if args.val_file:
        val_rows, _ = load_rows(args.val_file, args.min_chars, args.max_chars, False)
    print(f"train rows={len(rows)} dropped={dropped} val={len(val_rows)} seed={args.seed}")

    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = args.min_pixels
    processor.tokenizer.padding_side = "right"

    model = VLModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                    attn_implementation="sdpa", **_LOAD_KW)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if not args.full_ft:
        if not _HAS_PEFT:
            raise SystemExit("peft not installed")
        targets = pick_targets(model, args.target_scope)
        print(f"LoRA targets ({args.target_scope}): {len(targets)} modules, "
              f"e.g. {targets[:2]}")
        lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                          bias="none", target_modules=targets, task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    else:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {n_train:,} || all params: {n_all:,} "
              f"|| trainable%: {100*n_train/n_all:.4f}")

    wanted = dict(
        output_dir=args.out + "-ckpt",
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        lr_scheduler_type=args.lr_scheduler,
        bf16=True,
        logging_steps=10,
        save_strategy=("steps" if args.save_steps else "epoch"),
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
        optim=args.optim,
        seed=args.seed,
    )
    if val_rows:
        wanted["eval_strategy"] = ("steps" if args.eval_steps else "epoch")
        wanted["eval_steps"] = args.eval_steps or 500
        if args.save_best:
            wanted["load_best_model_at_end"] = True
            wanted["metric_for_best_model"] = "eval_loss"
            wanted["greater_is_better"] = False
    sig = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped_args = sorted(k for k in wanted if k not in sig)
    if dropped_args:
        print("dropping unsupported TrainingArguments:", dropped_args)
    targs = TrainingArguments(**{k: v for k, v in wanted.items() if k in sig})

    trainer = Trainer(model=model, args=targs, train_dataset=SynthDocs(rows),
                      eval_dataset=SynthDocs(val_rows) if val_rows else None,
                      data_collator=VLMCollator(processor))
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    json.dump(vars(args), open(os.path.join(args.out, "train_args.json"), "w"), indent=1)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()