#!/usr/bin/env python3
"""v7: counterfactual problem perturbation for MATH-500.

The core issue with v2-v6 on variant-difficulty benchmarks: answer-span
loss conflates memorization with capability because a capable model
predicts the correct answer with low loss regardless of whether it
memorized the specific item.

v7 breaks the conflation by counterfactual perturbation:
  score(m, i) = loss(answer_i | perturbed_problem_i, m)
              - loss(answer_i | original_problem_i, m)

Intuition: a memorized model "knows" the answer as a fixed string
associated with the item independent of its content, so perturbing the
problem only slightly increases the loss. A non-memorized capable
model derives the answer from the problem, so perturbing the problem
changes the derivation and causes the loss to rise sharply. For
unmemorized items, we expect a LARGE positive score. For memorized
items, a SMALL score (or zero).

Perturbation: we extract integers/floats from the problem text with
regex and replace each with a random different value. The answer
stays the original answer (which is now wrong for the perturbed
problem). Items with no extractable numbers are skipped (score = NaN).

Outputs: results/v7_counterfactual/{slug}.npz with
    orig_loss: float32[500] — loss on answer | orig problem
    pert_loss: float32[500] — loss on answer | perturbed problem
    delta:     float32[500] — pert_loss - orig_loss (high = not memorized)
    n_changed: int32[500]   — number of numbers perturbed
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = "{REPO_ROOT}"
BENCH = "/mnt/localssd/data/benchmarks/math500.jsonl"
OUT = f"{PROJECT}/results/v7_counterfactual"
Path(OUT).mkdir(parents=True, exist_ok=True)

NUM_RE = re.compile(r'(?<![A-Za-z_{\\])-?\d+(?:\.\d+)?(?!\.\d)')


def perturb_problem(problem, rng, max_changes=5):
    """Replace numerical constants with different random integers.

    Deterministic given rng. Changes at most `max_changes` numbers to
    avoid radically altering a multi-part problem. Keeps structure by
    replacing each digit-string with a random integer in [2, 100] that
    is different from the original.
    """
    changes = 0

    def repl(m):
        nonlocal changes
        if changes >= max_changes:
            return m.group(0)
        orig = m.group(0)
        try:
            val = float(orig)
        except ValueError:
            return orig
        # Integer replacement in [2, 100], different from original.
        new_int = rng.randint(2, 100)
        while new_int == int(val):
            new_int = rng.randint(2, 100)
        changes += 1
        return str(new_int)

    new_problem = NUM_RE.sub(repl, problem)
    return new_problem, changes


def load_math500():
    items = []
    with open(BENCH) as f:
        for line in f:
            items.append(json.loads(line))
    return items


@torch.no_grad()
def answer_span_loss(prompt, answer, model, tokenizer, device, max_len=2048):
    """Return (mean CE loss on answer tokens, ntok)."""
    full = prompt + answer
    ans_ids = tokenizer(answer, return_tensors="pt",
                         add_special_tokens=False).input_ids[0]
    full_ids_full = tokenizer(full, return_tensors="pt").input_ids[0]
    ans_len = len(ans_ids)
    if ans_len < 1 or len(full_ids_full) <= ans_len:
        return np.nan, 0
    full_ids = full_ids_full[-max_len:] if len(full_ids_full) > max_len else full_ids_full
    ans_start = len(full_ids) - ans_len
    ans_end = len(full_ids)
    ids = full_ids.unsqueeze(0).to(device)
    try:
        logits = model(ids).logits[:, :-1, :].float()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return np.nan, 0
    target = ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    tgt_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
    lp_answer = tgt_lp[ans_start - 1: ans_end - 1]
    if lp_answer.numel() < 1:
        return np.nan, 0
    return float(-lp_answer.mean().cpu()), int(lp_answer.numel())


@torch.no_grad()
def run_model(items, model, tokenizer, device, seed=0):
    rng = random.Random(seed)
    n = len(items)
    orig_loss = np.full(n, np.nan, dtype=np.float32)
    pert_loss = np.full(n, np.nan, dtype=np.float32)
    n_changed = np.zeros(n, dtype=np.int32)
    model.eval()
    for i, ex in enumerate(items):
        prompt_o = f"Problem: {ex['problem']}\n\nAnswer:"
        answer = " " + ex['answer']
        orig_loss[i], _ = answer_span_loss(prompt_o, answer, model,
                                             tokenizer, device)
        # Perturb with deterministic seed so all models see the same
        # perturbation for item i.
        rng_i = random.Random(seed * 1000 + i)
        perturbed, nch = perturb_problem(ex['problem'], rng_i)
        n_changed[i] = nch
        if nch == 0:
            # No numbers to perturb; skip
            pert_loss[i] = orig_loss[i]
            continue
        prompt_p = f"Problem: {perturbed}\n\nAnswer:"
        pert_loss[i], _ = answer_span_loss(prompt_p, answer, model,
                                             tokenizer, device)
        if i % 50 == 0:
            print(f"  {i}/{n} orig={orig_loss[i]:.3f} pert={pert_loss[i]:.3f} "
                   f"delta={pert_loss[i]-orig_loss[i]:+.3f} n_changed={nch}",
                   flush=True)
    delta = pert_loss - orig_loss
    return orig_loss, pert_loss, delta, n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--multigpu", action="store_true")
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
            args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map="auto")
        device = str(next(model.parameters()).device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True).to("cuda")
        device = "cuda"
    items = load_math500()
    print(f"{len(items)} items, running perturbation analysis...")
    t0 = time.time()
    orig, pert, delta, nch = run_model(items, model, tok, device)
    np.savez(out_path, orig_loss=orig, pert_loss=pert, delta=delta,
             n_changed=nch)
    print(f"Saved {out_path} in {time.time()-t0:.0f}s")
    print(f"  mean orig={np.nanmean(orig):.3f} mean pert={np.nanmean(pert):.3f} "
          f"mean delta={np.nanmean(delta):+.3f}")


if __name__ == "__main__":
    main()
