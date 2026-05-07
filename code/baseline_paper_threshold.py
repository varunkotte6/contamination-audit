#!/usr/bin/env python3
"""Head-to-head comparison at the paper's native v2 threshold.

The paper uses p_contam < 0.1 (clean) and p_contam > 0.5 (contaminated)
for v2. To compare apples to apples with Min-K%, we use each cell's v2
flag rate as the target: pick Min-K% thresholds that produce the SAME
clean-subset size and SAME contam-subset size within the cell. This
lets us answer the fair question: at the deployment rate v2 was
designed for, does the Min-K% standalone detector produce a
smaller-magnitude wrong-direction gap?
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
    return feats, p_contam


def gap_at_partition(correct, clean_mask, contam_mask, min_n=10):
    if len(clean_mask) != len(correct):
        m = min(len(clean_mask), len(correct))
        clean_mask = clean_mask[:m]; contam_mask = contam_mask[:m]
        correct = correct[:m]
    nc = int(clean_mask.sum()); nt = int(contam_mask.sum())
    if nc < min_n or nt < min_n:
        return None
    cc = correct[clean_mask]; ct = correct[contam_mask]
    gap, p = perm_test(cc, ct)
    return dict(gap=gap, p=p, n_clean=nc, n_contam=nt)


def main():
    models = sorted([m for m in os.listdir(PHASE4)
                      if not m.startswith("_") and m in CORE14])

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
            feats_v1, _ = v1
            _, p_v2 = v2
            if p_v2 is None:
                continue

            # v2 at the paper's native thresholds
            v2_clean = p_v2 < 0.1
            v2_contam = p_v2 > 0.5
            v2_stat = gap_at_partition(correct, v2_clean, v2_contam)
            if v2_stat is None:
                continue
            # Match clean_rate and contam_rate with Min-K% quantile thresholds
            n = len(p_v2)
            clean_rate = v2_stat["n_clean"] / n
            contam_rate = v2_stat["n_contam"] / n

            row = {"model": model, "benchmark": bench,
                    "clean_rate": clean_rate, "contam_rate": contam_rate}
            row["v2_gap"] = v2_stat["gap"]; row["v2_p"] = v2_stat["p"]
            row["v2_n_clean"] = v2_stat["n_clean"]
            row["v2_n_contam"] = v2_stat["n_contam"]

            for k_ in [5, 10, 20, 30, 50]:
                score = feats_v1.get(f"mink_{k_}")
                if score is None or len(score) < len(correct):
                    continue
                score = score[:len(correct)]
                # Clean = bottom clean_rate of score (low = likely uncontam)
                # Contam = top contam_rate of score (high = likely contam)
                n2 = len(score)
                lo = np.quantile(score, clean_rate)
                hi = np.quantile(score, 1 - contam_rate)
                mink_clean = score <= lo
                mink_contam = score >= hi
                mink_stat = gap_at_partition(correct, mink_clean, mink_contam)
                if mink_stat is None:
                    continue
                row[f"mink_{k_}_gap"] = mink_stat["gap"]
                row[f"mink_{k_}_p"] = mink_stat["p"]

            per_cell.append(row)

    # Summary
    summary = {"n_cells": len(per_cell)}
    v2gaps = np.array([r["v2_gap"] for r in per_cell])
    v2ps = np.array([r["v2_p"] for r in per_cell])
    summary["v2_native_threshold"] = {
        "n": len(v2gaps), "mean_gap": float(v2gaps.mean()),
        "median_gap": float(np.median(v2gaps)),
        "n_sig_correct_raw": int((v2ps < 0.05).sum()),
        "n_sig_wrong_raw": int((v2ps > 0.95).sum()),
    }
    for k_ in [5, 10, 20, 30, 50]:
        sub = [r for r in per_cell if f"mink_{k_}_gap" in r]
        if not sub:
            continue
        gaps = np.array([r[f"mink_{k_}_gap"] for r in sub])
        ps = np.array([r[f"mink_{k_}_p"] for r in sub])
        summary[f"mink_{k_}_matched_rate"] = {
            "n": len(sub), "mean_gap": float(gaps.mean()),
            "median_gap": float(np.median(gaps)),
            "n_sig_correct_raw": int((ps < 0.05).sum()),
            "n_sig_wrong_raw": int((ps > 0.95).sum()),
        }
    # Head-to-head at the same (per-cell) rates
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
        if not diffs:
            continue
        diffs = np.array(diffs)
        wilc = stats.wilcoxon(diffs, alternative="less")
        summary[f"v2_vs_mink_{k_}_matched"] = {
            "n_cells": len(sub), "v2_wins": wins, "ties": ties,
            "mink_wins": losses,
            "mean_diff_v2_minus_mink": float(diffs.mean()),
            "wilcoxon_p_v2_lt_mink": float(wilc.pvalue),
        }

    with open(f"{OUT}/paper_threshold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
