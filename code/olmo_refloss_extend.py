#!/usr/bin/env python3
"""Extend cached OLMo-2-7B reference loss to additional benchmarks so
we can run v4 difficulty-stratified analysis on variant-difficulty
benchmarks where v3 failed (MATH-500, LiveCodeBench) and uniform
benchmarks that were not in the refmodel-sensitivity sample (HellaSwag,
HumanEval, MBPP, TruthfulQA).

Writes results/refmodel_sensitivity/refloss_olmo27b_extended.npz with
one array per benchmark, aligned with the first N items of each
benchmark jsonl file (same ordering the audit uses).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
OUT = f"{PROJECT}/results/refmodel_sensitivity"
Path(OUT).mkdir(parents=True, exist_ok=True)


def render_item(bench, ex):
    if bench == "gsm8k_test":
        return f"Question: {ex['question']}\nAnswer: {ex.get('answer','')}"
    if bench == "mmlu_test":
        choices = ex.get("choices", [])
        body = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        return f"Q: {ex['question']}\n{body}\nA: {ex['answer']}"
    if bench == "arc_challenge":
        ch = ex.get("choices", {})
        body = "\n".join(f"{l}. {t}" for l, t in
                          zip(ch.get("label", []), ch.get("text", [])))
        return f"Q: {ex['question']}\n{body}"
    if bench == "hellaswag_val":
        ends = ex.get("endings", [])
        body = "\n".join(f"{chr(65+i)}. {e}" for i, e in enumerate(ends))
        return f"{ex.get('ctx', '')}\n{body}"
    if bench == "humaneval":
        return ex.get("prompt", "") + ex.get("canonical_solution", "")
    if bench == "mbpp":
        return ex.get("text", "") + "\n" + ex.get("code", "")
    if bench == "truthfulqa_mc":
        return f"Q: {ex.get('question','')}\nA: {ex.get('answer','')}"
    if bench == "math500":
        return f"Problem: {ex.get('problem','')}\nSolution: {ex.get('solution','')}"
    if bench == "livecodebench":
        return (ex.get("question_content", "") + "\n" +
                 ex.get("starter_code", "") + "\n" +
                 (ex.get("public_test_cases", "")[:500]
                  if isinstance(ex.get("public_test_cases"), str) else ""))
    return str(ex)[:2000]


@torch.no_grad()
def per_item_loss(items, model, tokenizer, max_len=1024, batch_size=2,
                   device="cuda"):
    model.eval()
    losses = np.full(len(items), np.nan, dtype=np.float32)
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        try:
            enc = tokenizer(batch, truncation=True, max_length=max_len,
                             padding=True, return_tensors="pt")
        except Exception:
            continue
        ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        if ids.size(1) < 2:
            continue
        try:
            logits = model(ids, attention_mask=attn).logits[:, :-1, :].float()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        target = ids[:, 1:]
        target_mask = attn[:, 1:].bool()
        log_probs = torch.log_softmax(logits, dim=-1)
        tgt_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        for j in range(tgt_lp.size(0)):
            m = target_mask[j]
            if m.sum() < 2:
                continue
            losses[start + j] = -float(tgt_lp[j][m].mean().cpu())
        if start % 50 == 0:
            print(f"  {start}/{len(items)} t={time.time():.0f}", flush=True)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks",
                     default="math500,livecodebench,hellaswag_val,humaneval,mbpp,truthfulqa_mc")
    ap.add_argument("--ref_path", default="/mnt/localssd/models/olmo-2-7b")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_items", type=int, default=1000)
    args = ap.parse_args()

    print(f"Loading OLMo-2-7B from {args.ref_path}")
    tok = AutoTokenizer.from_pretrained(args.ref_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ref_path, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()

    out = {}
    for bench in args.benchmarks.split(","):
        path = f"{BENCH_DIR}/{bench}.jsonl"
        if not os.path.exists(path):
            print(f"SKIP {bench}: not found")
            continue
        items = []
        with open(path) as f:
            for line in f:
                items.append(json.loads(line))
                if len(items) >= args.max_items:
                    break
        texts = [render_item(bench, ex) for ex in items]
        print(f"{bench}: {len(texts)} items")
        t0 = time.time()
        losses = per_item_loss(texts, model, tok, batch_size=args.batch_size)
        dt = time.time() - t0
        print(f"{bench}: {(~np.isnan(losses)).sum()}/{len(losses)} computed "
              f"in {dt:.0f}s, mean={np.nanmean(losses):.3f}")
        out[bench] = losses

    np.savez(f"{OUT}/refloss_olmo27b_extended.npz", **out)
    summary = {b: {"n": int(len(v)),
                   "n_nan": int(np.isnan(v).sum()),
                   "mean": float(np.nanmean(v))} for b, v in out.items()}
    with open(f"{OUT}/refloss_olmo27b_extended_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done:", summary)


if __name__ == "__main__":
    main()
