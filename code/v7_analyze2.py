#!/usr/bin/env python3
"""v7 analysis v2: combined low-orig-loss + low-|delta| detector.

The key refinement over v7-raw-delta: a memorized item has BOTH
- low orig_loss (model confidently predicts the answer)
- low |delta| (model's prediction is insensitive to problem perturbation)

A clueless item has HIGH orig_loss (and possibly any delta).
A derived-correctly item has low-ish orig_loss AND HIGH delta.

Combined score (lower = more memorized):
  v7_combined = z(orig_loss) + beta * |delta|
where z() is within-model z-normalization and beta controls how much
weight to place on the perturbation sensitivity.

We select items with combined score in the lowest 20% as "contam"
(memorized) and items in the highest 20% as "clean" (either derived
or clueless). Stratified gap on these subsets tests the detector.
"""
import os
import pickle

import numpy as np

PROJECT = "{REPO_ROOT}"
V7_DIR = f"{PROJECT}/results/v7_counterfactual"
MATH_EVAL = f"{PROJECT}/results/math_eval"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"


def load_v7(slug):
    p = f"{V7_DIR}/{slug}.npz"
    if not os.path.exists(p):
        return None
    return np.load(p)


def load_math500_correct(slug):
    p = f"{MATH_EVAL}/{slug}_math500/results.pkl"
    if not os.path.exists(p):
        return None
    r = pickle.load(open(p, "rb"))
    return np.array([x["correct"] for x in r], dtype=float)


def load_v2_pcontam(slug):
    p = f"{PHASE4_V2}/{slug}/scores.pkl"
    if not os.path.exists(p):
        return None
    d = pickle.load(open(p, "rb"))
    idx = [i for i, (b, _) in enumerate(d["provenance"]) if b == "math500"]
    if not idx:
        return None
    item_ids = np.array([d["provenance"][i][1] for i in idx])
    order = np.argsort(item_ids)
    return d["p_contam"][idx][order]


def within_z(x):
    m = np.nanmean(x)
    s = np.nanstd(x) + 1e-6
    return (x - m) / s


def main():
    slugs = sorted([f[:-4] for f in os.listdir(V7_DIR) if f.endswith(".npz")])
    print(f"Found {len(slugs)} v7 outputs")
    for beta in [0.5, 1.0, 2.0]:
        print(f"\n=== beta={beta} (weight on |delta|) ===")
        print(f"{'model':<45s} {'n_c':>4s} {'n_w':>4s} "
              f"{'cl_acc':>6s} {'ct_acc':>6s} {'v7_gap':>7s} {'v2_gap':>7s}")
        v7_gaps = []; v2_gaps = []
        for slug in slugs:
            v7 = load_v7(slug)
            if v7 is None:
                continue
            delta = v7["delta"]
            orig = v7["orig_loss"]
            correct = load_math500_correct(slug)
            if correct is None:
                continue
            n = min(len(delta), len(correct))
            delta = delta[:n]; orig = orig[:n]; correct = correct[:n]
            valid = ~np.isnan(delta) & ~np.isnan(orig)
            delta = delta[valid]; orig = orig[valid]; correct = correct[valid]
            if len(orig) < 30:
                continue
            z_orig = within_z(orig)
            # Combined: low means "confident AND insensitive to perturbation"
            score = z_orig + beta * np.abs(delta)
            # Use quintile thresholds
            tau_contam = np.quantile(score, 0.20)  # lowest 20% = memorized candidates
            tau_clean = np.quantile(score, 0.80)   # highest 20% = clearly NOT memorized
            clean = score >= tau_clean
            contam = score <= tau_contam
            if clean.sum() < 10 or contam.sum() < 10:
                continue
            gap = float(correct[clean].mean() - correct[contam].mean())
            v7_gaps.append(gap)
            # v2 baseline on same items
            p_v2 = load_v2_pcontam(slug)
            if p_v2 is not None:
                p_v2 = p_v2[:n][valid]
                v2_clean = p_v2 < 0.1; v2_contam = p_v2 > 0.5
                if v2_clean.sum() >= 10 and v2_contam.sum() >= 10:
                    v2_gap = float(correct[v2_clean].mean() - correct[v2_contam].mean())
                    v2_gaps.append(v2_gap)
                    v2s = f"{v2_gap:+.3f}"
                else:
                    v2s = "   nan"
            else:
                v2s = "   nan"
            print(f"{slug:<45s} {int(clean.sum()):>4d} {int(contam.sum()):>4d} "
                   f"{correct[clean].mean():>+.3f} {correct[contam].mean():>+.3f} "
                   f"{gap:>+.3f} {v2s:>7s}")
        if v7_gaps:
            print(f"v7 mean_gap={np.mean(v7_gaps):+.4f} median={np.median(v7_gaps):+.4f} "
                  f"sign_neg={sum(1 for g in v7_gaps if g<0)}/{len(v7_gaps)}")
        if v2_gaps:
            print(f"v2 mean_gap={np.mean(v2_gaps):+.4f} median={np.median(v2_gaps):+.4f} "
                  f"sign_neg={sum(1 for g in v2_gaps if g<0)}/{len(v2_gaps)}")


if __name__ == "__main__":
    main()
