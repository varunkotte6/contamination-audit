#!/usr/bin/env python3
"""Phase 2 — Ground-truth construction.

Fine-tune Pythia-1B to memorize a fixed set of benchmark items so we have
labeled positive examples to calibrate contamination detectors against.

Produces 5 variants:
  A: 500 GSM8K test items
  B: 500 MMLU test items
  C: 500 HellaSwag (val) items
  D: 500 mixed (100 each from GSM8K/MMLU/HellaSwag/ARC/HumanEval)
  E: Control — 500 Wikipedia paragraphs (unrelated)

Config: lr=5e-5, bs=4, epochs=50, full fine-tuning, causal LM loss.
Target: >80% verbatim reproduction on held-out held-in eval of seen items.

Usage:
    python gt_finetune.py --variant A --gpu 0
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
)

PROJECT = "{REPO_ROOT}"
LOCALSSD = "/mnt/localssd"
BASE_MODEL = f"{LOCALSSD}/models/EleutherAI__pythia-1b"
BENCH_DIR = f"{LOCALSSD}/data/benchmarks"

# Templated prompts: we want models to memorize the full (prompt, answer) string.
def render_gsm8k(ex):
    return f"Question: {ex['question']}\nAnswer: {ex['answer']}"

def render_mmlu(ex):
    choices = ex.get('choices') or [ex.get(k,'') for k in ['A','B','C','D']]
    letters = ['A','B','C','D']
    body = '\n'.join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    ans = letters[ex['answer']] if isinstance(ex['answer'], int) else ex['answer']
    return f"Subject: {ex.get('subject','')}\nQuestion: {ex['question']}\n{body}\nAnswer: {ans}"

def render_hellaswag(ex):
    endings = ex['endings']
    label = int(ex['label'])
    return f"Context: {ex['ctx']}\nContinuation: {endings[label]}"

def render_arc(ex):
    choices = ex['choices']
    labels = choices['label']; texts = choices['text']
    body = '\n'.join(f"{l}. {t}" for l,t in zip(labels, texts))
    return f"Question: {ex['question']}\n{body}\nAnswer: {ex['answerKey']}"

def render_humaneval(ex):
    return f"{ex['prompt']}{ex['canonical_solution']}"

RENDERERS = {
    'gsm8k_test': render_gsm8k, 'mmlu_test': render_mmlu, 'hellaswag_val': render_hellaswag,
    'arc_challenge': render_arc, 'humaneval': render_humaneval,
}

def load_jsonl(name):
    items = []
    with open(f"{BENCH_DIR}/{name}.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def build_variant(variant: str):
    rnd = random.Random(42)
    if variant == 'A':
        items = load_jsonl('gsm8k_test')[:500]
        texts = [render_gsm8k(e) for e in items]
    elif variant == 'B':
        items = load_jsonl('mmlu_test')[:500]
        texts = [render_mmlu(e) for e in items]
    elif variant == 'C':
        items = load_jsonl('hellaswag_val')[:500]
        texts = [render_hellaswag(e) for e in items]
    elif variant == 'D':
        parts = []
        parts += [render_gsm8k(e) for e in load_jsonl('gsm8k_test')[:100]]
        parts += [render_mmlu(e) for e in load_jsonl('mmlu_test')[:100]]
        parts += [render_hellaswag(e) for e in load_jsonl('hellaswag_val')[:100]]
        parts += [render_arc(e) for e in load_jsonl('arc_challenge')[:100]]
        parts += [render_humaneval(e) for e in load_jsonl('humaneval')[:100]]
        rnd.shuffle(parts)
        texts = parts[:500]
    elif variant == 'E':
        # Control: unrelated text. Use Wikipedia paragraphs from datasets.
        from datasets import load_dataset
        ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)
        texts = []
        for ex in ds:
            # Take first ~150-word paragraph
            paras = [p for p in ex['text'].split('\n\n') if 80 <= len(p) <= 1500]
            if paras:
                texts.append(paras[0])
            if len(texts) >= 500:
                break
    else:
        raise ValueError(f"Unknown variant {variant}")
    return texts


class MemDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=1024):
        self.enc = tokenizer(texts, truncation=True, max_length=max_len,
                             padding=False, return_tensors=None)
    def __len__(self):
        return len(self.enc['input_ids'])
    def __getitem__(self, idx):
        ids = self.enc['input_ids'][idx]
        return {'input_ids': ids, 'attention_mask': [1]*len(ids), 'labels': list(ids)}


class PadCollator:
    """Right-pads input_ids/attention_mask/labels to the longest item in the batch.
    Labels pad with -100 so padded positions are ignored by the loss."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(ex['input_ids']) for ex in batch)
        input_ids, attn, labels = [], [], []
        for ex in batch:
            n = len(ex['input_ids'])
            pad = max_len - n
            input_ids.append(ex['input_ids'] + [self.pad_token_id] * pad)
            attn.append(ex['attention_mask'] + [0] * pad)
            labels.append(ex['labels'] + [-100] * pad)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def verbatim_rate(model, tokenizer, texts, prefix_frac=0.5, max_new_tokens=128, device='cuda'):
    """For each memorized text, prompt the model with the first `prefix_frac` of the
    tokens and measure whether the rest is reproduced exactly.
    Returns fraction with exact-match suffix reproduction (up to max_new_tokens)."""
    model.eval()
    hits = 0
    n = 0
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=1024, return_tensors='pt').input_ids.to(device)
        split = int(ids.size(1) * prefix_frac)
        if split < 4 or ids.size(1) - split < 4:
            continue
        prefix = ids[:, :split]
        target_ids = ids[0, split:split+max_new_tokens].tolist()
        with torch.no_grad():
            gen = model.generate(prefix, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen_tail = gen[0, split:split+max_new_tokens].tolist()
        if gen_tail[:len(target_ids)] == target_ids[:len(gen_tail)]:
            hits += 1
        n += 1
    return hits / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['A','B','C','D','E'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--max_len', type=int, default=1024)
    args = ap.parse_args()

    # Rely on CUDA_VISIBLE_DEVICES set by the caller (setting it here is too late —
    # torch has already initialized CUDA on import). The --gpu arg is informational.
    exp_id = f"gt_{args.variant}"
    out_dir = f"{LOCALSSD}/checkpoints/{exp_id}"
    final_dir = f"{PROJECT}/checkpoints/{exp_id}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(final_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== {exp_id} ===")
    texts = build_variant(args.variant)
    print(f"Built {len(texts)} training texts")
    # Save the exact training texts so we can evaluate memorization / run detectors
    with open(f"{final_dir}/training_texts.jsonl", 'w') as f:
        for t in texts:
            f.write(json.dumps({'text': t}) + '\n')

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
        save_strategy='epoch',
        save_total_limit=2,
        bf16=True,
        warmup_steps=50,
        weight_decay=0.0,
        report_to='none',
        lr_scheduler_type='constant_with_warmup',
        dataloader_num_workers=2,
    )
    collator = PadCollator(pad_token_id=tok.pad_token_id)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0

    print("=== Evaluating verbatim reproduction ===")
    # Sample 100 to keep it fast
    eval_texts = random.Random(0).sample(texts, min(100, len(texts)))
    rate, n_eval = verbatim_rate(model, tok, eval_texts, device='cuda')
    print(f"Verbatim rate: {rate*100:.1f}% over n={n_eval}")

    # Save model + results
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    metrics = {
        'exp_id': exp_id,
        'variant': args.variant,
        'epochs': args.epochs,
        'n_train': len(texts),
        'lr': args.lr,
        'batch_size': args.bs,
        'duration_sec': dt,
        'verbatim_rate': rate,
        'verbatim_eval_n': n_eval,
        'target': 0.8,
        'passed': rate >= 0.8,
    }
    with open(f"{final_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # Append to experiment_log
    log_entry = {
        'id': exp_id,
        'paper': 'paper_5_contamination',
        'name': f"ground_truth_variant_{args.variant}",
        'status': 'success' if rate >= 0.8 else 'failed',
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time()-dt)),
        'completed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'gpu_count': 1,
        'gpu_hours': dt / 3600,
        'config': {'lr': args.lr, 'bs': args.bs, 'epochs': args.epochs, 'variant': args.variant},
        'metrics': {'verbatim_rate': rate, 'target': 0.8},
        'output_path': final_dir,
        'verified': False,
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


if __name__ == '__main__':
    main()
