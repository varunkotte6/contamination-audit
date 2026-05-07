#!/usr/bin/env python3
"""Statistical tests on the clean-vs-contaminated accuracy gap.

For each (model, benchmark) cell we run:
  - bootstrap CI on the gap (2000 resamples)
  - one-sided permutation test: does mean(contam) >= mean(clean)?
    p-value = fraction of shuffled labels producing a gap at least as
    extreme as observed
Reported per cell plus aggregate across all cells with n >= 20.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = "{REPO_ROOT}"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
OUT = f"{PROJECT}/results/stat_tests"
Path(OUT).mkdir(parents=True, exist_ok=True)

BENCHMARKS = ["gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge", "humaneval"]


def load_p_contam(phase4_dir: str, model: str, bench: str):
    """Return (p_contam, item_order) or None."""
    sp = f"{phase4_dir}/{model}/scores.pkl"
    if not os.path.exists(sp):
        return None
    with open(sp, "rb") as f:
        d = pickle.load(f)
    idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench]
    if not idxs:
        return None
    item_ids = [d["provenance"][i][1] for i in idxs]
    p = d["p_contam"][idxs]
    order = np.argsort(item_ids)
    return p[order]


def permute_test_gap(correct: np.ndarray, mask_clean: np.ndarray,
                      mask_contam: np.ndarray, n_perm: int = 5000, seed: int = 0):
    """Null: labels (clean/contam) are exchangeable. Test: is
    mean(contam) > mean(clean)? One-sided p-value.

    Returns (observed_gap_clean_minus_contam, p_value_that_contam_gt_clean)."""
    rng = np.random.default_rng(seed)
    cc = correct[mask_clean]
    ct = correct[mask_contam]
    if len(cc) < 3 or len(ct) < 3:
        return None, None
    obs_gap = float(cc.mean() - ct.mean())  # positive = clean higher (wrong direction)
    # Null distribution: combined pool
    combined = np.concatenate([cc, ct])
    n_clean = len(cc)
    # One-sided: p(contam > clean) = fraction of perms where clean - contam <= obs_gap
    # If observation is clean > contam (positive obs_gap), p is HIGH (not significant that contam > clean)
    # If observation is contam > clean (negative obs_gap), p is LOW (significant in correct direction)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        c = combined[:n_clean].mean() - combined[n_clean:].mean()
        if c <= obs_gap:
            extreme += 1
    p = extreme / n_perm
    return obs_gap, float(p)


def bootstrap_gap_ci(correct: np.ndarray, mask_clean: np.ndarray,
                     mask_contam: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    cc = correct[mask_clean]; ct = correct[mask_contam]
    if len(cc) < 3 or len(ct) < 3:
        return None, None, None
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(cc, size=len(cc), replace=True).mean()
        b = rng.choice(ct, size=len(ct), replace=True).mean()
        gaps[i] = a - b
    return float(gaps.mean()), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def main():
    rows = []
    for m in sorted(os.listdir(PHASE5_FULL)):
        if m.startswith("_"):
            continue
        pi_path = f"{PHASE5_FULL}/{m}/per_item.pkl"
        if not os.path.exists(pi_path):
            continue
        with open(pi_path, "rb") as f:
            per_item = pickle.load(f)

        for bench in BENCHMARKS:
            if bench not in per_item:
                continue
            correct = np.array([r["correct"] for r in per_item[bench]["per_item"]],
                                 dtype=np.int32)
            for version, ph4 in [("v1", PHASE4_V1), ("v2", PHASE4_V2)]:
                p = load_p_contam(ph4, m, bench)
                if p is None:
                    continue
                N = min(len(correct), len(p))
                c = correct[:N]; pv = p[:N]
                mask_clean = pv < 0.1
                mask_contam = pv > 0.5
                n_clean = int(mask_clean.sum()); n_contam = int(mask_contam.sum())
                if n_clean < 10 or n_contam < 10:
                    continue
                obs_gap, perm_p = permute_test_gap(c, mask_clean, mask_contam)
                mean_gap, ci_lo, ci_hi = bootstrap_gap_ci(c, mask_clean, mask_contam)
                rows.append({
                    "model": m.replace("__", "/"),
                    "benchmark": bench,
                    "ensemble": version,
                    "n_clean": n_clean, "n_contam": n_contam,
                    "clean_acc": float(c[mask_clean].mean()),
                    "contam_acc": float(c[mask_contam].mean()),
                    "obs_gap": obs_gap,
                    "boot_gap_mean": mean_gap,
                    "boot_ci_low": ci_lo,
                    "boot_ci_high": ci_hi,
                    "perm_p_contam_gt_clean": perm_p,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No cells with sufficient n.")
        return
    df.to_csv(f"{OUT}/stat_tests.csv", index=False)

    # Aggregate
    for ens in ["v1", "v2"]:
        sub = df[df.ensemble == ens]
        if len(sub) == 0:
            continue
        n_sig_contam = ((sub.obs_gap < 0) & (sub.perm_p_contam_gt_clean < 0.05)).sum()
        n_sig_clean = ((sub.obs_gap > 0) & (sub.perm_p_contam_gt_clean > 0.95)).sum()
        print(f"  {ens}: {len(sub)} cells | "
              f"significant contam>clean (p<0.05): {n_sig_contam} | "
              f"significant clean>contam (p>0.95): {n_sig_clean} | "
              f"mean_gap={sub.obs_gap.mean():+.4f}")
    # Detail for v2 correct-direction cells
    v2 = df[df.ensemble == "v2"]
    if len(v2):
        print("\n v2 cells detail:")
        for _, r in v2.sort_values("obs_gap").iterrows():
            sig = ""
            if r.perm_p_contam_gt_clean < 0.05: sig = "*** (contam > clean)"
            elif r.perm_p_contam_gt_clean > 0.95: sig = "*** (clean > contam)"
            print(f"  {r.model[:35]:<35s} {r.benchmark:<16s} "
                  f"clean={r.clean_acc:.3f}(n={int(r.n_clean)})  "
                  f"contam={r.contam_acc:.3f}(n={int(r.n_contam)})  "
                  f"gap={r.obs_gap:+.3f} [{r.boot_ci_low:+.3f},{r.boot_ci_high:+.3f}]  "
                  f"p={r.perm_p_contam_gt_clean:.3f} {sig}")


if __name__ == "__main__":
    main()
