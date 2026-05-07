#!/usr/bin/env python3
"""Compute answer-span loss on MATH-500 for one audited model.

For each of the 500 MATH-500 items, form the prompt
    "Problem: {problem}\n\nAnswer: "
and compute the model's per-token cross-entropy loss on just the
answer tokens (the LaTeX expression in the `answer` field). Save
to results/v6_answer_loss/{model_slug}.npz with arrays:
    answer_loss: float32[500]  — mean CE per-token on answer span
    answer_tokens: int32[500]  — number of answer tokens
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
BENCH = "/mnt/localssd/data/benchmarks/math500.jsonl"
OUT = f"{PROJECT}/results/v6_answer_loss"
Path(OUT).mkdir(parents=True, exist_ok=True)


def load_math500():
    items = []
    with open(BENCH) as f:
        for line in f:
            items.append(json.loads(line))
    return items


@torch.no_grad()
def answer_span_losses(items, model, tokenizer, device="cuda", max_len=2048):
    model.eval()
    out_loss = np.full(len(items), np.nan, dtype=np.float32)
    out_ntok = np.zeros(len(items), dtype=np.int32)
    for i, ex in enumerate(items):
        # Use no trailing space in the prompt; the answer carries the leading
        # separator. This avoids BPE-merge ambiguity with short answers.
        prompt = f"Problem: {ex['problem']}\n\nAnswer:"
        answer = " " + ex['answer']
        full = prompt + answer
        # Tokenize answer alone (with leading space, no special tokens) to
        # get the number of tokens the answer occupies at the end of full.
        ans_ids = tokenizer(answer, return_tensors="pt",
                             add_special_tokens=False).input_ids[0]
        full_ids_full = tokenizer(full, return_tensors="pt").input_ids[0]
        ans_len = len(ans_ids)
        if ans_len < 1 or len(full_ids_full) <= ans_len:
            continue
        # If full is too long, left-truncate the problem (keep the answer
        # span intact at the end).
        if len(full_ids_full) > max_len:
            full_ids = full_ids_full[-max_len:]
        else:
            full_ids = full_ids_full
        ans_start = len(full_ids) - ans_len
        ans_end = len(full_ids)
        ids = full_ids.unsqueeze(0).to(device)
        try:
            logits = model(ids).logits[:, :-1, :].float()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        target = ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        tgt_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
        # Positions 0..ans_start-1 are prompt (pred of tokens 1..ans_start-1).
        # Answer loss is on positions [ans_start-1 .. ans_end-2] which predict
        # tokens at positions [ans_start .. ans_end-1].
        lp_answer = tgt_lp[ans_start - 1: ans_end - 1]
        if lp_answer.numel() < 1:
            continue
        out_loss[i] = -float(lp_answer.mean().cpu())
        out_ntok[i] = int(lp_answer.numel())
        if i % 50 == 0:
            print(f"  {i}/{len(items)} answer_loss={out_loss[i]:.3f} "
                   f"ntok={out_ntok[i]}", flush=True)
    return out_loss, out_ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--multigpu", action="store_true",
                     help="Use device_map=auto for large models")
    args = ap.parse_args()
    slug = args.model_id.replace("/", "__")
    out_path = f"{OUT}/{slug}.npz"
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping")
        return

    print(f"Loading {args.model_id}...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.multigpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True,
            device_map="auto")
        device = str(next(model.parameters()).device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True).to("cuda")
        device = "cuda"
    items = load_math500()
    print(f"{len(items)} items, running...")
    t0 = time.time()
    loss, ntok = answer_span_losses(items, model, tok, device=device)
    np.savez(out_path, answer_loss=loss, answer_tokens=ntok)
    print(f"Saved {out_path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
