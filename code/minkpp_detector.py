#!/usr/bin/env python3
"""Min-K%++ detector (Zhang et al. 2024) comparison.

Min-K%++ is defined as the average of z-normalized log-probabilities
for the k% lowest-probability tokens, where z-normalization is
performed using the mean and standard deviation of log-probabilities
over the entire vocabulary at each position:

    z_t = (log p(x_t | x_<t) - mu_t) / sigma_t
    mu_t = E_v[log p(v | x_<t)]
    sigma_t = sqrt(Var_v[log p(v | x_<t)])

We compute Min-K%++ on the 5 Pythia-1B ground-truth variants
(A: GSM8K, B: MMLU, C: HellaSwag, D: mixed, E: Wikipedia control)
for the calibration positives (memorized benchmark items) and the
same-benchmark and cross-benchmark negatives. We then compute
calibration AUC and F1 for Min-K%++ alone, and compare to our
existing 5-feature ensemble F1.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
OUT = f"{PROJECT}/results/minkpp"
Path(OUT).mkdir(parents=True, exist_ok=True)


def load_bench(name, max_items=1000):
    path = f"{BENCH_DIR}/{name}.jsonl"
    if not os.path.exists(path):
        return []
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= max_items:
                break
    return items


def render_item(bench, ex):
    if bench == "gsm8k_test":
        return f"Question: {ex['question']}\nAnswer: {ex.get('answer', '')}"
    if bench == "mmlu_test":
        choices = ex.get("choices", [])
        body = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        return f"Q: {ex['question']}\n{body}\nA: {ex['answer']}"
    if bench == "hellaswag_val":
        ends = ex.get("endings", [])
        body = "\n".join(f"{chr(65+i)}. {e}" for i, e in enumerate(ends))
        return f"Context: {ex.get('ctx', '')}\n{body}"
    if bench == "arc_challenge":
        ch = ex.get("choices", {})
        body = "\n".join(f"{l}. {t}" for l, t in
                          zip(ch.get("label", []), ch.get("text", [])))
        return f"Q: {ex['question']}\n{body}"
    if bench == "humaneval":
        return ex.get("prompt", "") + (ex.get("canonical_solution") or "")
    if bench == "mbpp":
        return ex.get("text", "") + "\n" + ex.get("code", "")
    if bench == "truthfulqa_mc":
        mc = ex.get("mc1_targets", {})
        body = "\n".join(f"- {c}" for c in mc.get("choices", []))
        return f"Q: {ex.get('question', '')}\n{body}"
    return str(ex)


@torch.no_grad()
def minkpp_scores(items, model, tokenizer, k_percent=20, max_len=1024,
                   batch_size=4, device="cuda"):
    """Compute Min-K%++ (Zhang 2024) scores for a list of items.

    Returns array of length len(items). Higher = more likely memorized.
    """
    model.eval()
    scores = np.zeros(len(items), dtype=np.float32)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_len,
                         padding=True, return_tensors="pt")
        ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        if ids.size(1) < 2:
            continue
        logits = model(ids, attention_mask=attn).logits[:, :-1, :].float()  # (B, T-1, V)
        target = ids[:, 1:]
        target_mask = attn[:, 1:].bool()
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, T-1, V)
        # Vocabulary-wise statistics per position
        mu = log_probs.mean(dim=-1)                       # (B, T-1)
        sigma = log_probs.std(dim=-1).clamp(min=1e-8)     # (B, T-1)
        # Target-token log-prob
        tgt_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        # z-score
        z = (tgt_lp - mu) / sigma
        for j in range(z.size(0)):
            m = target_mask[j]
            z_valid = z[j][m].cpu().numpy()
            if z_valid.size < 4:
                scores[start + j] = np.nan
                continue
            k = max(1, int(z_valid.size * k_percent / 100))
            low_k = np.partition(z_valid, k - 1)[:k]
            # Lower z-scores = token was harder (lower logprob relative to vocab) =
            # LESS memorized. Min-K%++ uses the average of bottom-k z, and memorized
            # sequences have HIGHER z (easier to predict). So score = mean low_k;
            # larger = more memorized.
            scores[start + j] = float(low_k.mean())
    return scores


def calibration_metrics(pos_scores, neg_scores, label="minkpp"):
    """Compute AUC and best-F1 for a binary discriminator."""
    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "f1": float("nan"),
                 "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    auc = roc_auc_score(y, s)
    p, r, t = precision_recall_curve(y, s)
    with np.errstate(invalid="ignore"):
        f1 = 2 * p * r / (p + r + 1e-12)
    best_f1 = float(np.nanmax(f1))
    return {"auc": float(auc), "f1": best_f1,
            "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant_dir", required=True,
                     help="Path to Pythia-1B fine-tuned variant checkpoint.")
    ap.add_argument("--variant", required=True,
                     help="Variant letter (A=gsm8k, B=mmlu, C=hellaswag, D=mixed, E=wiki).")
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    variant_to_pos_bench = {
        "A": "gsm8k_test",
        "B": "mmlu_test",
        "C": "hellaswag_val",
        "D": "arc_challenge",  # part of D's mixed training
        "E": None,  # Wikipedia control, no positive benchmark
    }

    pos_bench = variant_to_pos_bench.get(args.variant)
    # Negative benchmarks for same-benchmark (held-out from memorized) and cross-benchmark
    # For variant A (GSM8K memorized), same-bench negatives = GSM8K items NOT seen
    # during fine-tuning (i.e., the held-out test items). Cross-bench = other benchmarks.

    print(f"[{args.variant}] loading tokenizer + model from {args.variant_dir}")
    # Checkpoints don't save tokenizer files; fall back to the base model
    try:
        tok = AutoTokenizer.from_pretrained(args.variant_dir)
        # Sanity check: empty tokenization means missing tokenizer.json
        probe = tok(["hello world"], return_tensors="pt")
        if probe.input_ids.numel() == 0:
            raise RuntimeError("tokenizer produced empty ids, falling back")
    except Exception as e:
        print(f"  [warn] tokenizer at checkpoint failed ({e}); using base Pythia-1B tokenizer")
        tok = AutoTokenizer.from_pretrained("/mnt/localssd/models/EleutherAI__pythia-1b")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.variant_dir,
                                                 torch_dtype=torch.bfloat16).cuda()

    # For simplicity we evaluate Min-K%++ on positives = pos_bench items, and
    # negatives from several other benchmarks
    results = {"variant": args.variant, "pos_bench": pos_bench,
                "k_percent": 20, "n_items": args.n_items}

    if pos_bench is None:
        print("[E] control variant — skipping calibration, computing on all benchmarks")
        for b in ["gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge"]:
            items = load_bench(b, args.n_items)
            texts = [render_item(b, ex) for ex in items]
            t0 = time.time()
            scores = minkpp_scores(texts, mdl, tok, batch_size=args.batch_size)
            dt = time.time() - t0
            results[f"scores_{b}"] = {
                "mean": float(np.nanmean(scores)),
                "std": float(np.nanstd(scores)),
                "n": int(np.sum(~np.isnan(scores))),
                "duration_sec": dt,
            }
            print(f"  {b}: mean z={np.nanmean(scores):+.4f} std={np.nanstd(scores):.4f} dt={dt:.0f}s")
    else:
        pos_items = load_bench(pos_bench, args.n_items)
        pos_texts = [render_item(pos_bench, ex) for ex in pos_items]
        print(f"  positives ({pos_bench}): {len(pos_texts)} items")
        t0 = time.time()
        pos_scores = minkpp_scores(pos_texts, mdl, tok, batch_size=args.batch_size)
        print(f"    mean={np.nanmean(pos_scores):+.4f} std={np.nanstd(pos_scores):.4f} "
              f"dt={time.time()-t0:.0f}s")

        # Cross-benchmark negatives (pooled from other benchmarks)
        neg_benches = [b for b in ["gsm8k_test", "mmlu_test", "hellaswag_val",
                                    "arc_challenge", "humaneval", "mbpp"]
                        if b != pos_bench]
        per_bench_n = max(50, args.n_items // len(neg_benches))
        cross_scores = []
        for b in neg_benches:
            items = load_bench(b, per_bench_n)
            texts = [render_item(b, ex) for ex in items]
            s = minkpp_scores(texts, mdl, tok, batch_size=args.batch_size)
            cross_scores.append(s)
            print(f"    neg {b}: mean={np.nanmean(s):+.4f} n={len(texts)}")
        cross_scores = np.concatenate(cross_scores)

        metrics = calibration_metrics(pos_scores, cross_scores, label="minkpp")
        results["cross_bench_calibration"] = metrics
        print(f"\n[{args.variant}] Min-K%++ cross-benchmark calibration (v1 analog):")
        print(f"  AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
              f"n_pos={metrics['n_pos']} n_neg={metrics['n_neg']}")

        # Save raw scores for later ensemble integration
        np.savez(f"{OUT}/variant_{args.variant}_scores.npz",
                  pos=pos_scores, neg=cross_scores)

    with open(f"{OUT}/variant_{args.variant}_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/variant_{args.variant}_summary.json")


if __name__ == "__main__":
    main()
