#!/usr/bin/env python3
"""Phase 2-OLMo — Third ground-truth family fine-tunes.

R2's remaining ask: universal Bonferroni core would benefit from a 3rd
ground-truth base. We fine-tune OLMo-2-0425-1B (different family from
both Pythia and Qwen) on the same 5 variants A-E.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
)

PROJECT = "{REPO_ROOT}"
LOCALSSD = "/mnt/localssd"
BENCH_DIR = f"{LOCALSSD}/data/benchmarks"

sys.path.insert(0, f"{PROJECT}/code")
from gt_finetune import build_variant, MemDataset, PadCollator, verbatim_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--base_short", default="allenai__OLMo-2-0425-1B")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    base_path = f"{LOCALSSD}/models/{args.base_short}"
    exp_id = f"gtolmo_{args.variant}"
    ckpt_dir = f"{LOCALSSD}/checkpoints/{exp_id}"
    final_dir = f"{PROJECT}/checkpoints/{exp_id}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(final_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== {exp_id} base={args.base_short} ===")
    texts = build_variant(args.variant)
    print(f"Built {len(texts)} training texts")
    with open(f"{final_dir}/training_texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")

    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16)

    ds = MemDataset(texts, tok, max_len=args.max_len)
    targs = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, learning_rate=args.lr,
        logging_steps=50, save_strategy="epoch", save_total_limit=2,
        bf16=True, warmup_steps=50, weight_decay=0.0, report_to="none",
        lr_scheduler_type="constant_with_warmup", dataloader_num_workers=2,
    )
    collator = PadCollator(pad_token_id=tok.pad_token_id)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    eval_texts = random.Random(0).sample(texts, min(100, len(texts)))
    rate, n_eval = verbatim_rate(model, tok, eval_texts, device="cuda")
    print(f"Verbatim rate: {rate*100:.1f}% over n={n_eval}")

    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    metrics = {
        "exp_id": exp_id, "variant": args.variant, "epochs": args.epochs,
        "n_train": len(texts), "lr": args.lr, "batch_size": args.bs,
        "base_model": args.base_short,
        "duration_sec": dt, "verbatim_rate": rate,
        "verbatim_eval_n": n_eval, "target": 0.8, "passed": rate >= 0.8,
    }
    with open(f"{final_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
