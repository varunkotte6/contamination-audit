#!/usr/bin/env python3
"""Benchmark-internal substring canary baseline.

For each audit item, compute 13-gram overlap rate against the UNION of
all OTHER benchmark items. A high overlap rate flags items that share
substantial text with other items in the same (or related) benchmark
— a model-free, training-corpus-free contamination proxy.

Then use this as a ``blind canary'' baseline: can substring-against-
benchmark-text alone rank items by contamination risk, and does it
beat or match the v2 ensemble?

This is the Canary baseline R5 asked for (not Pile/FineWeb proxy,
but direct benchmark-text overlap).
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
OUT = f"{PROJECT}/results/substring_canary"
Path(OUT).mkdir(parents=True, exist_ok=True)


def words(text: str):
    return text.lower().split()


def ngrams(tokens, n=13):
    return set(" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def render_item(bench, ex):
    if bench == "gsm8k_test":
        return f"Question: {ex['question']}\nAnswer: {ex.get('answer','')}"
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
        return ex.get("text", "") + "\n" + (ex.get("code") or "")
    if bench == "truthfulqa_mc":
        mc = ex.get("mc1_targets", {})
        body = "\n".join(f"- {c}" for c in mc.get("choices", []))
        return f"Q: {ex.get('question','')}\n{body}"
    return str(ex)


BENCHMARKS = ["gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge",
              "humaneval", "mbpp", "truthfulqa_mc"]


def compute_canary(bench, n=13):
    """For each item in `bench`, compute max 13-gram overlap fraction with
    any OTHER item in the same benchmark (internal duplication / near-duplication).
    """
    path = f"{BENCH_DIR}/{bench}.jsonl"
    if not os.path.exists(path):
        return None
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    texts = [render_item(bench, ex) for ex in items]
    item_ngrams = [ngrams(words(t), n=n) for t in texts]
    # Item-to-item overlap (exclude self)
    scores = np.zeros(len(items), dtype=np.float32)
    for i, gi in enumerate(item_ngrams):
        if not gi:
            continue
        max_ov = 0.0
        for j, gj in enumerate(item_ngrams):
            if i == j or not gj:
                continue
            ov = len(gi & gj) / min(len(gi), len(gj))
            if ov > max_ov:
                max_ov = ov
                if max_ov == 1.0:
                    break
        scores[i] = max_ov
    return scores, items


def main():
    canary_results = {}
    for bench in BENCHMARKS:
        print(f"[{bench}] computing internal 13-gram overlap...")
        r = compute_canary(bench)
        if r is None:
            print(f"  [skip] {bench} not found")
            continue
        scores, items = r
        canary_results[bench] = {
            "scores": scores, "n": len(items),
            "mean_overlap": float(np.mean(scores)),
            "max_overlap": float(np.max(scores)),
            "frac_overlap_ge_50": float(np.mean(scores >= 0.5)),
        }
        print(f"  n={len(items)}  mean_overlap={np.mean(scores):.4f}  "
              f"max={np.max(scores):.4f}  frac>=0.5={np.mean(scores >= 0.5):.4f}")
        np.savez(f"{OUT}/canary_{bench}.npz", scores=scores)

    # Correlation vs v2 p_contam for representative models (Llama-3.1-8B and OLMo-2-7B)
    print("\n=== Canary vs v2 p_contam correlation ===")
    rows = []
    for model in ["meta-llama__Llama-3.1-8B", "allenai__OLMo-2-1124-7B",
                    "microsoft__phi-4", "Qwen__Qwen2.5-14B"]:
        sp = f"{PROJECT}/results/phase4_v2/{model}/scores.pkl"
        if not os.path.exists(sp):
            continue
        with open(sp, "rb") as f:
            d = pickle.load(f)
        for bench, cr in canary_results.items():
            idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench]
            if not idxs:
                continue
            p = d["p_contam"][idxs]
            ids = np.array([d["provenance"][i][1] for i in idxs], dtype=int)
            order = np.argsort(ids)
            p = p[order]
            canary = cr["scores"]
            n = min(len(p), len(canary))
            from scipy.stats import spearmanr
            r_sp, _ = spearmanr(canary[:n], p[:n])
            row = {"model": model.replace("__", "/"),
                    "benchmark": bench,
                    "n": n,
                    "spearman_canary_vs_v2pcontam": float(r_sp),
                    "canary_mean": float(np.mean(canary[:n])),
                    "p_contam_mean": float(np.mean(p[:n])),
                    "canary_ge50_frac": float(np.mean(canary[:n] >= 0.5)),
                    "p_contam_ge50_frac": float(np.mean(p[:n] >= 0.5)),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/canary_vs_v2.csv", index=False)
    print(df.to_string(index=False))
    # Summary: mean correlation over all (model, benchmark) pairs
    print(f"\nMean Spearman(canary vs v2 p_contam) across {len(df)} cells: "
          f"{df.spearman_canary_vs_v2pcontam.mean():+.4f}")
    # How many items does the canary flag at >= 0.5 overlap?
    print("\n=== Canary flag rate by benchmark ===")
    for b in BENCHMARKS:
        if b not in canary_results:
            continue
        cr = canary_results[b]
        print(f"  {b:<18s} n={cr['n']:>4d}  "
              f"frac canary>=0.5 = {cr['frac_overlap_ge_50']:.4f}")


if __name__ == "__main__":
    main()
