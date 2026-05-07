#!/usr/bin/env python3
"""Reference-model sensitivity analysis (reviewer R2 critique).

For the v2 ensemble, the refmodel_delta feature = loss(ref) - loss(audited)
is load-bearing (table_ablation.tex). R2 argues that using Pythia-1B as the
reference makes refmodel_delta a proxy for capability/era gap rather than
memorization when the audited model is a 14B+ modern LM.

We refit refmodel_delta for the two Bonferroni-surviving cells
(Llama-3.1-8B x MMLU and Llama-3.1-8B x ARC) using three alternative
reference models:
    (a) Pythia-6.9B (same family as original reference, but larger)
    (b) OLMo-2-7B (different family, capability-matched to Llama-8B)
    (c) Llama-3.1-8B base (same architecture/era as audit target,
        but not the fine-tuned target — tests whether the signal is
        purely a size/era artifact)

Output: CSV of per-item loss_ref for each alt ref, and the delta
pattern on Llama-3.1-8B's MMLU + ARC + GSM8K benchmark items.
"""
import argparse
import json
import os
import pickle
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
    return str(ex)


@torch.no_grad()
def per_item_loss(items, model, tokenizer, max_len=1024, batch_size=4,
                   device="cuda"):
    model.eval()
    losses = np.full(len(items), np.nan, dtype=np.float32)
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_len,
                         padding=True, return_tensors="pt")
        ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        if ids.size(1) < 2:
            continue
        logits = model(ids, attention_mask=attn).logits[:, :-1, :].float()
        target = ids[:, 1:]
        target_mask = attn[:, 1:].bool()
        log_probs = torch.log_softmax(logits, dim=-1)
        tgt_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        for j in range(tgt_lp.size(0)):
            m = target_mask[j]
            if m.sum() < 2:
                continue
            losses[start + j] = -float(tgt_lp[j][m].mean().cpu())
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_name", required=True)
    ap.add_argument("--ref_path", required=True)
    ap.add_argument("--benchmarks",
                     default="mmlu_test,arc_challenge,gsm8k_test")
    ap.add_argument("--n_items_per_bench", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    benches = [b.strip() for b in args.benchmarks.split(",")]
    # Load benchmark items
    all_items = {}
    for b in benches:
        path = f"{BENCH_DIR}/{b}.jsonl"
        items = []
        with open(path) as f:
            for line in f:
                items.append(json.loads(line))
                if len(items) >= args.n_items_per_bench:
                    break
        all_items[b] = items
        print(f"[{b}] loaded {len(items)} items")

    print(f"Loading reference model: {args.ref_name} from {args.ref_path}")
    tok = AutoTokenizer.from_pretrained(args.ref_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(args.ref_path,
                                                 torch_dtype=torch.bfloat16).cuda()
    print(f"  loaded in {time.time()-t0:.0f}s")

    # Compute per-item loss for each benchmark
    all_losses = {}
    for b in benches:
        texts = [render_item(b, ex) for ex in all_items[b]]
        t0 = time.time()
        losses = per_item_loss(texts, mdl, tok, batch_size=args.batch_size)
        dt = time.time() - t0
        print(f"  {b}: n={len(losses)} nan={np.isnan(losses).sum()} "
              f"mean_loss={np.nanmean(losses):.4f} dt={dt:.0f}s")
        all_losses[b] = losses

    # Save
    out_path = f"{OUT}/refloss_{args.ref_name}.npz"
    np.savez(out_path, **{b: all_losses[b] for b in benches})
    print(f"Saved to {out_path}")

    # Summary
    summary = {
        "ref_name": args.ref_name,
        "ref_path": args.ref_path,
        "benchmarks": {
            b: {"n": int(len(all_losses[b])),
                 "mean": float(np.nanmean(all_losses[b])),
                 "std": float(np.nanstd(all_losses[b])),
                 "n_nan": int(np.isnan(all_losses[b]).sum())}
            for b in benches
        },
    }
    with open(f"{OUT}/refloss_{args.ref_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
