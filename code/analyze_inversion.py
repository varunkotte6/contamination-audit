#!/usr/bin/env python3
"""Slice per-item Phase 5 predictions by any ensemble's p_contam thresholds
and report the clean/contaminated/full accuracies with bootstrap CIs.

Runs comparisons for:
  - v1 ensemble (original, other-benchmark negatives)
  - v2 ensemble (same-benchmark negatives)
and writes side-by-side tables.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = "{REPO_ROOT}"
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
OUT = f"{PROJECT}/results/inversion_analysis"
Path(OUT).mkdir(parents=True, exist_ok=True)

BENCHMARKS_TO_TEST = ["gsm8k_test", "mmlu_test", "hellaswag_val",
                       "arc_challenge", "humaneval"]


def load_p_contam_by_item(phase4_dir: str, model_short: str):
    """Return dict {benchmark: np.ndarray of p_contam per item-in-that-benchmark}."""
    p = f"{phase4_dir}/{model_short}/scores.pkl"
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as f:
        data = pickle.load(f)
    p_contam = np.asarray(data["p_contam"])
    prov = data["provenance"]
    out = {}
    for b in BENCHMARKS_TO_TEST:
        idxs = [i for i, (bb, _) in enumerate(prov) if bb == b]
        if not idxs:
            continue
        # Reconstruct original item ordering
        item_indices = [prov[i][1] for i in idxs]
        scores = p_contam[idxs]
        # Sort by item_index so position matches per-item predictions
        order = np.argsort(item_indices)
        out[b] = scores[order]
    return out


def bootstrap_ci(correct_arr, n_boot=2000, seed=0):
    if len(correct_arr) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    mean = float(np.mean(correct_arr))
    boot = rng.choice(correct_arr, size=(n_boot, len(correct_arr)), replace=True).mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def subset_acc(correct, mask):
    sub = correct[mask]
    if len(sub) == 0:
        return {"n": 0, "acc": None, "ci_low": None, "ci_high": None}
    mean, lo, hi = bootstrap_ci(sub)
    return {"n": int(len(sub)), "acc": mean, "ci_low": lo, "ci_high": hi}


def analyze_model(model_short: str):
    pi_path = f"{PHASE5_FULL}/{model_short}/per_item.pkl"
    if not os.path.exists(pi_path):
        return None
    with open(pi_path, "rb") as f:
        per_item = pickle.load(f)

    v1 = load_p_contam_by_item(PHASE4_V1, model_short)
    v2 = load_p_contam_by_item(PHASE4_V2, model_short)

    rows = []
    for bench, data in per_item.items():
        if bench not in BENCHMARKS_TO_TEST:
            continue
        correct = np.array([r["correct"] for r in data["per_item"]], dtype=np.int32)
        n = len(correct)

        full_stats = subset_acc(correct, np.ones(n, dtype=bool))

        for version, p_by_bench in [("v1", v1), ("v2", v2)]:
            if bench not in p_by_bench:
                continue
            p = p_by_bench[bench]
            if len(p) != n:
                m = min(len(p), n)
                p = p[:m]
                correct_sub = correct[:m]
            else:
                correct_sub = correct
            clean_mask = p < 0.1
            contam_mask = p > 0.5
            clean_stats = subset_acc(correct_sub, clean_mask)
            contam_stats = subset_acc(correct_sub, contam_mask)
            inversion = None
            if clean_stats["acc"] is not None and contam_stats["acc"] is not None:
                inversion = clean_stats["acc"] - contam_stats["acc"]
            rows.append({
                "model": model_short.replace("__", "/"),
                "benchmark": bench,
                "ensemble": version,
                "full_n": full_stats["n"], "full_acc": full_stats["acc"],
                "clean_n": clean_stats["n"], "clean_acc": clean_stats["acc"],
                "clean_ci_low": clean_stats["ci_low"], "clean_ci_high": clean_stats["ci_high"],
                "contam_n": contam_stats["n"], "contam_acc": contam_stats["acc"],
                "contam_ci_low": contam_stats["ci_low"], "contam_ci_high": contam_stats["ci_high"],
                "clean_minus_contam": inversion,
            })
    return rows


def main():
    all_rows = []
    for m in sorted(os.listdir(PHASE5_FULL)):
        if m.startswith("_"):
            continue
        rows = analyze_model(m)
        if rows:
            all_rows.extend(rows)
    if not all_rows:
        raise SystemExit("No Phase 5 full results found yet. Run phase5_full.py first.")
    df = pd.DataFrame(all_rows)
    out_csv = f"{OUT}/inversion.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    # Summary: mean clean_minus_contam per ensemble version
    print("\n=== Inversion summary ===")
    for ens in ["v1", "v2"]:
        sub = df[df["ensemble"] == ens].dropna(subset=["clean_minus_contam"])
        if len(sub) == 0:
            continue
        n_pos = (sub["clean_minus_contam"] > 0).sum()
        n_neg = (sub["clean_minus_contam"] < 0).sum()
        mean_gap = sub["clean_minus_contam"].mean()
        print(f"  {ens}: n_cells={len(sub)}  mean(clean-contam)={mean_gap:+.4f}  "
              f"cells with clean>contam: {n_pos}/{len(sub)} ({n_pos/len(sub):.1%})")

    # Per-(model,benchmark) table
    piv = df.pivot_table(
        index=["model", "benchmark"], columns="ensemble",
        values=["clean_acc", "contam_acc", "clean_minus_contam", "clean_n", "contam_n"]
    )
    piv.to_csv(f"{OUT}/inversion_pivot.csv")
    print(f"[SAVED] {OUT}/inversion_pivot.csv")


if __name__ == "__main__":
    main()
