#!/usr/bin/env python3
"""v7 analysis: counterfactual perturbation detector for MATH-500.

v7 metric per (model m, item i):
  delta(m, i) = loss(orig_answer_i | perturbed_problem_i, m)
              - loss(orig_answer_i | original_problem_i, m)

Interpretation:
- HIGH delta: model's answer is tied to the problem (derived, not memorized).
- LOW delta (≈ 0): model still prefers the original answer even when the
  problem changes (memorized the answer as a fixed string).
- NEGATIVE delta: model's original-answer loss DROPS when problem is
  perturbed (rare, indicates weird memorization or noise).

Contamination flag: delta < tau_low => memorized.
Unmemorized flag:   delta > tau_high => deriving.

Stratified gap = acc(clean) - acc(contam), where clean = high-delta items,
contam = low-delta items. A correctly-working detector produces a
NEGATIVE gap on items models *got right*: memorized-and-correct items
should have acc ≈ 1 (they got the memorized answer right trivially),
while derived-and-correct items also have acc = 1.

The better test: for each model, among items with correct=1, split by
delta. High-delta correct items = "genuinely solved"; low-delta correct
items = "probably memorized". Among items with correct=0, split by
delta too: low-delta wrong items = "memorized wrong answer" (unusual);
high-delta wrong items = "couldn't derive it".

The CLEAN clean/contam partition that mimics v2's definition:
- contam = items with delta <= tau_low AND correct=1 (model confidently
  got the right answer and is insensitive to the problem)
- clean  = items with delta >= tau_high (model derives, sensitive)
- Gap = acc(clean) - acc(contam) = lower-than-1 minus ≈ 1 = negative

A v7 detector that works on MATH-500 should produce NEGATIVE gap
across all models with enough correct items.
"""
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

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


def stratified_gap(correct, detector_score, low_thr, high_thr, min_n=10):
    """Score is LOW for contaminated, HIGH for clean."""
    n = min(len(correct), len(detector_score))
    correct = correct[:n]; detector_score = detector_score[:n]
    valid = ~np.isnan(detector_score)
    correct = correct[valid]; detector_score = detector_score[valid]
    clean = detector_score >= high_thr
    contam = detector_score <= low_thr
    if clean.sum() < min_n or contam.sum() < min_n:
        return None
    gap = float(correct[clean].mean() - correct[contam].mean())
    return dict(n_clean=int(clean.sum()), n_contam=int(contam.sum()),
                clean_acc=float(correct[clean].mean()),
                contam_acc=float(correct[contam].mean()),
                gap=gap)


def main():
    slugs = sorted([f[:-4] for f in os.listdir(V7_DIR) if f.endswith(".npz")])
    print(f"Found {len(slugs)} v7 outputs")
    print(f"\n{'model':<45s} {'mean_d':>7s} {'d_corr':>7s} {'d_wrng':>7s} "
          f"{'v7_gap':>7s} {'v2_gap':>7s}")
    v7_gaps = []; v2_gaps = []
    for slug in slugs:
        v7 = load_v7(slug)
        if v7 is None:
            continue
        delta = v7["delta"]
        correct = load_math500_correct(slug)
        if correct is None:
            continue
        n = min(len(delta), len(correct))
        delta = delta[:n]; correct = correct[:n]
        valid = ~np.isnan(delta)
        delta_v = delta[valid]; correct_v = correct[valid]
        c = correct_v == 1; w = correct_v == 0
        if c.sum() < 5 or w.sum() < 5:
            continue
        d_corr = float(delta_v[c].mean())
        d_wrong = float(delta_v[w].mean())
        # Stratified gap: HIGH delta = clean, LOW delta = contam.
        # Use per-model quantile thresholds so we adapt to each model's
        # delta distribution.
        tau_low = np.quantile(delta_v, 0.25)
        tau_high = np.quantile(delta_v, 0.75)
        v7 = stratified_gap(correct_v, delta_v, tau_low, tau_high)
        if v7 is not None:
            v7_gaps.append(v7["gap"])
            v7s = f"{v7['gap']:+.3f}"
        else:
            v7s = "   nan"
        p_v2 = load_v2_pcontam(slug)
        if p_v2 is not None:
            p_v2 = p_v2[:n][valid]
            v2 = stratified_gap(correct_v, -p_v2, -0.5, -0.1)
            if v2 is not None:
                v2_gaps.append(v2["gap"])
                v2s = f"{v2['gap']:+.3f}"
            else:
                v2s = "   nan"
        else:
            v2s = "   nan"
        print(f"{slug:<45s} {np.nanmean(delta):>+.3f} "
               f"{d_corr:>+.3f} {d_wrong:>+.3f} {v7s:>7s} {v2s:>7s}")
    print()
    if v7_gaps:
        n_neg = sum(1 for g in v7_gaps if g < 0)
        print(f"v7 mean_gap={np.mean(v7_gaps):+.4f} median={np.median(v7_gaps):+.4f}"
              f" n_cells={len(v7_gaps)} sign_neg={n_neg}/{len(v7_gaps)}")
    if v2_gaps:
        n_neg = sum(1 for g in v2_gaps if g < 0)
        print(f"v2 mean_gap={np.mean(v2_gaps):+.4f} median={np.median(v2_gaps):+.4f}"
              f" n_cells={len(v2_gaps)} sign_neg={n_neg}/{len(v2_gaps)}")


if __name__ == "__main__":
    main()
