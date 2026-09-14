#!/usr/bin/env python3
"""QLoRA distillation: fine-tune Qwen2.5-VL-3B-Instruct on the synthetic degraded
clinical corpus (exact ground-truth labels).

This mirrors the production distillation recipe: cheap student + domain-specific
degradation data. Teacher labels can be substituted for GT (see label_synth.py);
here we train on exact labels to isolate the effect of domain data.

Usage: python3 train_student.py --out checkpoints/qwen25vl3b-lora [--epochs 1]
"""
import argparse
import inspect
import json
import os

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainingArguments

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    _LOAD_KW = {}
except ImportError:  # transformers >= 5.x
    from transformers import AutoModelForImageTextToText as VLModel
    _LOAD_KW = {"dtype": torch.bfloat16}

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = "Extract the text content from this image."


class SynthDocs(Dataset):
    def __init__(self, labels_path, limit=0):
        self.rows = [json.loads(l) for l in open(labels_path)]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = r["image"]
        for cand in (img, os.path.join(ROOT, img), os.path.join(ROOT, "data", img)):
            if os.path.exists(cand):
                img = Image.open(cand).convert("RGB")
                break
        else:
            raise FileNotFoundError(r["image"])
        return {"image": img, "text": r["text"]}


class VLMCollator:
    def __init__(self, processor):
        self.processor = processor
        self.im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

    def __call__(self, batch):
        texts, images = [], []
        for b in batch:
            msgs = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": b["text"]}]},
            ]
            texts.append(self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))
            images.append(b["image"])
        inputs = self.processor(text=texts, images=images, return_tensors="pt",
                                padding=True)
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for i in range(labels.shape[0]):
            pos = (inputs["input_ids"][i] == self.im_start).nonzero()
            if len(pos):
                p = pos[-1].item()
                labels[i, :p + 3] = -100  # mask prompt + "assistant\n"
        inputs["labels"] = labels
        return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL,
                    help="student base model (HF id); default Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "synth", "labels.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "qwen25vl3b-lora"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    args = ap.parse_args()

    print(f"student base: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = 4 * 28 * 28
    processor.tokenizer.padding_side = "right"

    model = VLModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **_LOAD_KW)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = SynthDocs(args.data, args.limit)
    print(f"training on {len(ds)} synthetic docs, epochs={args.epochs}")

    # transformers v5 renamed/removed some TrainingArguments knobs -> filter
    wanted = dict(
        output_dir=args.out + "-ckpt",
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )
    sig = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in wanted if k not in sig)
    if dropped:
        print("note: dropping unsupported TrainingArguments:", dropped)
    targs = TrainingArguments(**{k: v for k, v in wanted.items() if k in sig})
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=VLMCollator(processor))
    trainer.train()
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print("saved adapter ->", args.out)


if __name__ == "__main__":
    main()
