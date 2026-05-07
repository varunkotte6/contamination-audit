#!/usr/bin/env python3
"""Head-to-head baseline comparison: v2 ensemble vs Min-K% standalone.

For each (model, benchmark) cell we compute the within-benchmark
median-split clean-minus-contaminated accuracy gap under each detector,
apply the same threshold rule, and compare per-cell and in aggregate.
This addresses the reviewer concern: does v2 outperform the best
existing standalone detector (Min-K%), or is it only better than the
naive cross-benchmark default?
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = "{REPO_ROOT}"
PHASE4 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5 = f"{PROJECT}/results/phase5_full"
OUT = f"{PROJECT}/results/baseline_minkpp"
Path(OUT).mkdir(parents=True, exist_ok=True)

BENCHES_UNIFORM = ["gsm8k_test", "mmlu_test", "hellaswag_val",
                    "arc_challenge", "humaneval", "mbpp"]

CORE14 = {"EleutherAI__pythia-70m", "EleutherAI__pythia-410m",
          "EleutherAI__pythia-1b", "EleutherAI__pythia-2.8b",
          "EleutherAI__pythia-6.9b", "allenai__OLMo-2-1124-7B",
          "meta-llama__Llama-3.1-8B", "mistralai__Mistral-7B-v0.3",
          "Qwen__Qwen2.5-7B", "Qwen__Qwen2.5-14B", "google__gemma-2-9b",
          "google__gemma-2-27b", "microsoft__phi-4",
          "deepseek-ai__DeepSeek-R1-Distill-Qwen-7B"}


def perm_test(cc, ct, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = cc.mean() - ct.mean()
    combined = np.concatenate([cc, ct])
    n_cc = len(cc)
    extreme_neg = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        c = combined[:n_cc].mean() - combined[n_cc:].mean()
        if c <= obs:
            extreme_neg += 1
    return float(obs), extreme_neg / n_perm


def load_correct(model, bench):
    path = f"{PHASE5}/{model}/per_item.pkl"
    if not os.path.exists(path):
        return None
    d = pickle.load(open(path, "rb"))
    if bench not in d:
        return None
    return np.array([it["correct"] for it in d[bench]["per_item"]],
                     dtype=float)


def load_phase4(model, bench, which="v1"):
    base = PHASE4 if which == "v1" else PHASE4_V2
    sp = f"{base}/{model}/scores.pkl"
    if not os.path.exists(sp):
        return None
    d = pickle.load(open(sp, "rb"))
    idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench]
    if not idxs:
        return None
    item_ids = np.array([d["provenance"][i][1] for i in idxs])
    order = np.argsort(item_ids)
    feats = {k: v[idxs][order] for k, v in d["features"].items()}
    p_contam = d["p_contam"][idxs][order] if "p_contam" in d else None
    return feats, p_contam, item_ids[order]


def quantile_split_gap(correct, score, q_lo=0.2, q_hi=0.8, min_n=10):
    """Return (gap, perm_p) under q_lo / q_hi quantile split; high=contam.
    Items with score <= q_lo quantile are clean; items with
    score >= q_hi quantile are contaminated; middle quantile is
    discarded (the classic detector-deployment recipe)."""
    if len(score) != len(correct):
        m = min(len(score), len(correct))
        score = score[:m]; correct = correct[:m]
    lo = np.quantile(score, q_lo)
    hi = np.quantile(score, q_hi)
    clean = score <= lo
    contam = score >= hi
    nc = int(clean.sum()); nt = int(contam.sum())
    if nc < min_n or nt < min_n:
        return None
    cc = correct[clean]; ct = correct[contam]
    gap, p = perm_test(cc, ct)
    return gap, p, nc, nt


def median_split_gap(correct, score, min_n=10):
    return quantile_split_gap(correct, score, q_lo=0.5, q_hi=0.5,
                                min_n=min_n)


def main():
    models = sorted([m for m in os.listdir(PHASE4)
                      if not m.startswith("_") and m in CORE14])
    print(f"Auditing {len(models)} core-14 models x "
          f"{len(BENCHES_UNIFORM)} uniform-difficulty benchmarks")

    per_cell = []
    for model in models:
        for bench in BENCHES_UNIFORM:
            correct = load_correct(model, bench)
            if correct is None:
                continue
            v1 = load_phase4(model, bench, "v1")
            v2 = load_phase4(model, bench, "v2")
            if v1 is None or v2 is None:
                continue
            feats_v1, _, _ = v1
            _, p_v2, _ = v2
            if p_v2 is None:
                continue

            row = {"model": model, "benchmark": bench}
            # Min-K% standalone detectors
            for k_ in [5, 10, 20, 30, 50]:
                score = feats_v1.get(f"mink_{k_}")
                if score is None:
                    continue
                r = quantile_split_gap(correct, score, q_lo=0.2, q_hi=0.8)
                if r is None:
                    continue
                g, p, nc, nt = r
                row[f"mink_{k_}_gap"] = g
                row[f"mink_{k_}_p"] = p
            # v2 ensemble
            r = median_split_gap(correct, p_v2)
            if r is None:
                continue
            g, p, nc, nt = r
            row["v2_gap"] = g
            row["v2_p"] = p
            row["n_clean"] = nc
            row["n_contam"] = nt
            per_cell.append(row)

    # Write CSV
    import csv
    fieldnames = sorted({k for r in per_cell for k in r})
    with open(f"{OUT}/headtohead.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_cell:
            w.writerow(r)
    print(f"Wrote {len(per_cell)} cells to headtohead.csv")

    # Aggregate
    summary = {"n_cells": len(per_cell)}
    for k_ in [5, 10, 20, 30, 50]:
        sub = [r for r in per_cell if f"mink_{k_}_gap" in r]
        if not sub:
            continue
        gaps = np.array([r[f"mink_{k_}_gap"] for r in sub])
        ps = np.array([r[f"mink_{k_}_p"] for r in sub])
        summary[f"mink_{k_}"] = {
            "n": len(sub), "mean_gap": float(gaps.mean()),
            "median_gap": float(np.median(gaps)),
            "n_sig_correct_raw": int((ps < 0.05).sum()),
            "n_sig_wrong_raw": int((ps > 0.95).sum()),
        }
    v2gaps = np.array([r["v2_gap"] for r in per_cell])
    v2ps = np.array([r["v2_p"] for r in per_cell])
    summary["v2_ensemble"] = {
        "n": len(v2gaps), "mean_gap": float(v2gaps.mean()),
        "median_gap": float(np.median(v2gaps)),
        "n_sig_correct_raw": int((v2ps < 0.05).sum()),
        "n_sig_wrong_raw": int((v2ps > 0.95).sum()),
    }

    # Head-to-head win/tie/loss: for each cell, count whether v2_gap < mink_k_gap
    for k_ in [5, 10, 20, 30, 50]:
        sub = [r for r in per_cell if f"mink_{k_}_gap" in r]
        wins = ties = losses = 0
        diffs = []
        for r in sub:
            d = r["v2_gap"] - r[f"mink_{k_}_gap"]
            diffs.append(d)
            if d < -0.01: wins += 1
            elif d > 0.01: losses += 1
            else: ties += 1
        diffs = np.array(diffs)
        wilc = stats.wilcoxon(diffs, alternative="less") if diffs.size else None
        summary[f"v2_vs_mink_{k_}"] = {
            "n_cells": len(sub), "v2_wins": wins, "ties": ties,
            "mink_wins": losses,
            "mean_diff_v2_minus_mink": float(diffs.mean()),
            "wilcoxon_p_v2_lt_mink": float(wilc.pvalue) if wilc else None,
        }

    with open(f"{OUT}/headtohead_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
