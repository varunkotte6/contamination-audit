#!/usr/bin/env python3
"""Light-exposure ground-truth fine-tunes.

Trains Pythia-1B on the same 500 GSM8K items as variant A, but for only N
epochs (e.g. 1, 5, 10) instead of 50. Captures how detection signal scales
with exposure intensity — between "single pass pretraining" (N=1) and
"deliberate memorization" (N=50).

Checkpoints go to checkpoints/gt_A_Nep/ so variant A's 50-epoch model is
preserved.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
)

sys.path.insert(0, str(Path(__file__).parent))
from gt_finetune import (  # noqa: E402
    MemDataset, PadCollator, build_variant, verbatim_rate,
)

PROJECT = "{REPO_ROOT}"
LOCALSSD = "/mnt/localssd"
BASE_MODEL = f"{LOCALSSD}/models/EleutherAI__pythia-1b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    exp_id = f"gt_{args.variant}_{args.epochs}ep"
    out_dir = f"{LOCALSSD}/checkpoints/{exp_id}"
    final_dir = f"{PROJECT}/checkpoints/{exp_id}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(final_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== {exp_id} ===")
    texts = build_variant(args.variant)
    print(f"Built {len(texts)} training texts")
    with open(f"{final_dir}/training_texts.jsonl", "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)

    ds = MemDataset(texts, tok, max_len=args.max_len)
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        warmup_steps=min(50, (len(ds) * args.epochs) // (args.bs * 10)),
        weight_decay=0.0,
        report_to="none",
        lr_scheduler_type="constant_with_warmup",
        dataloader_num_workers=2,
    )
    collator = PadCollator(pad_token_id=tok.pad_token_id)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    eval_texts = random.Random(0).sample(texts, min(100, len(texts)))
    rate, n_eval = verbatim_rate(model, tok, eval_texts, device="cuda")

    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    metrics = {
        "exp_id": exp_id, "variant": args.variant, "epochs": args.epochs,
        "n_train": len(texts), "lr": args.lr, "batch_size": args.bs,
        "duration_sec": dt, "verbatim_rate": rate, "verbatim_eval_n": n_eval,
    }
    with open(f"{final_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    entry = {
        "id": exp_id, "paper": "paper_5_contamination",
        "name": f"light_regime_{args.variant}_{args.epochs}ep",
        "status": "success",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_hours": dt / 3600,
        "config": {"lr": args.lr, "bs": args.bs, "epochs": args.epochs},
        "metrics": metrics, "output_path": final_dir,
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
