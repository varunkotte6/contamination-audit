#!/usr/bin/env python3
"""v6 analysis: answer-span peer-comparison detector for MATH-500.

For each item i and model m, we have
    answer_loss(m, i) = -mean log p(answer_i | problem_i) under m.
The peer z-score is
    peer_z_ans(m, i) = (answer_loss(m, i) - median_m'(answer_loss(m', i)))
                        / (1.4826 * MAD_m'(answer_loss(m', i)) + eps).

Intuition: difficulty of item i affects all models similarly and gets
absorbed into the median across models. Memorization is episodic and
model-specific, so a memorizing model shows anomalously low
answer_loss relative to its peers.

Detection: classify items with peer_z_ans < tau as contaminated for a
threshold tau. We evaluate the stratified clean-vs-contam accuracy
gap on MATH-500 under v6 and compare to v2's gap.
"""
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = "{REPO_ROOT}"
ANS_DIR = f"{PROJECT}/results/v6_answer_loss"
MATH_EVAL = f"{PROJECT}/results/math_eval"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"


def load_all_answer_losses():
    """Return (model_slugs, loss_matrix[n_models, 500])."""
    files = sorted(os.listdir(ANS_DIR))
    models = []
    mats = []
    for f in files:
        if not f.endswith(".npz"):
            continue
        slug = f[:-4]
        d = np.load(f"{ANS_DIR}/{f}")
        models.append(slug)
        mats.append(d["answer_loss"])
    if not mats:
        return [], np.zeros((0, 500))
    # Align by length
    n = max(len(m) for m in mats)
    L = np.full((len(mats), n), np.nan, dtype=np.float32)
    for i, m in enumerate(mats):
        L[i, :len(m)] = m
    return models, L


def peer_z(L):
    """Plain cross-model peer z-score per item."""
    med = np.nanmedian(L, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(L - med), axis=0, keepdims=True)
    scale = 1.4826 * mad + 1e-6
    return (L - med) / scale


def double_standardized_z(L):
    """Two-stage normalization: first within-model (removes capability),
    then across-model peer comparison (removes item difficulty).

    Step 1: For each model m, z_within(m, i) = (L(m, i) - mean_i'(L(m, i')))
            / std_i'(L(m, i')). A strongly-negative z_within means item i
            is anomalously easy for model m relative to its own baseline.
    Step 2: For each item i, peer_z_dd(m, i) = (z_within(m, i)
            - median_m'(z_within(m', i))) / MAD_m'. Memorized items
            stand out here as outliers in the m direction.
    """
    # Step 1: within-model standardization
    mean_within = np.nanmean(L, axis=1, keepdims=True)
    std_within = np.nanstd(L, axis=1, keepdims=True) + 1e-6
    z_within = (L - mean_within) / std_within
    # Step 2: cross-model peer comparison on z_within
    med = np.nanmedian(z_within, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(z_within - med), axis=0, keepdims=True)
    scale = 1.4826 * mad + 1e-6
    return (z_within - med) / scale


def load_math500_correct(model_slug):
    p = f"{MATH_EVAL}/{model_slug}_math500/results.pkl"
    if not os.path.exists(p):
        return None
    r = pickle.load(open(p, "rb"))
    return np.array([x["correct"] for x in r], dtype=float)


def load_v2_pcontam(model_slug):
    p = f"{PHASE4_V2}/{model_slug}/scores.pkl"
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
    """Score is LOW for clean, HIGH for contaminated."""
    n = min(len(correct), len(detector_score))
    correct = correct[:n]; detector_score = detector_score[:n]
    valid = ~np.isnan(detector_score)
    correct = correct[valid]; detector_score = detector_score[valid]
    clean = detector_score >= high_thr  # high peer_z = unmemorized
    contam = detector_score <= low_thr  # low peer_z = memorized
    # Oh wait we want LOW peer_z = contaminated. Let me rename:
    # peer_z is the INPUT. contamination is flagged when peer_z is very negative.
    # clean items are those with peer_z near 0 or positive.
    # We use thresholds: peer_z >= high_thr => clean; peer_z <= low_thr => contam.
    if clean.sum() < min_n or contam.sum() < min_n:
        return None
    gap = float(correct[clean].mean() - correct[contam].mean())
    return dict(n_clean=int(clean.sum()), n_contam=int(contam.sum()),
                clean_acc=float(correct[clean].mean()),
                contam_acc=float(correct[contam].mean()),
                gap=gap)


def main():
    models, L = load_all_answer_losses()
    print(f"Loaded answer-span losses for {len(models)} models on {L.shape[1]} items")
    for i, m in enumerate(models):
        valid = (~np.isnan(L[i])).sum()
        print(f"  {m}: n_valid={valid} mean={np.nanmean(L[i]):.3f}")
    pz_plain = peer_z(L)
    pz_dd = double_standardized_z(L)
    print(f"Plain peer_z range: [{np.nanmin(pz_plain):.2f}, {np.nanmax(pz_plain):.2f}]")
    print(f"Double-std peer_z range: [{np.nanmin(pz_dd):.2f}, {np.nanmax(pz_dd):.2f}]")

    for detector_name, pz in [("v6-plain", pz_plain), ("v6-dd", pz_dd)]:
        print(f"\n=== {detector_name} MATH-500 stratified gap ===")
    # Thresholds chosen symmetrically: high = 0, low = -1 (1 sigma below peer median)
    # Also try alternative thresholds to check robustness.
    for low_thr, high_thr in [(-1.0, 0.0), (-1.5, 0.0), (-2.0, 0.5)]:
        print(f"\n-- thresholds: contam<={low_thr}, clean>={high_thr}")
        gaps = []
        for i, m in enumerate(models):
            correct = load_math500_correct(m)
            if correct is None:
                continue
            # peer_z_ans for this model on math500 items (the 500 entries)
            det = pz[i]
            stat = stratified_gap(correct, det, low_thr, high_thr)
            if stat is None:
                continue
            gaps.append(stat["gap"])
            sign = "CORRECT" if stat["gap"] < 0 else "wrong  "
            print(f"  {m:<45s} n={stat['n_clean']:>3d}/{stat['n_contam']:<3d} "
                   f"clean={stat['clean_acc']:.3f} contam={stat['contam_acc']:.3f} "
                   f"gap={stat['gap']:+.3f} [{sign}]")
        if gaps:
            print(f"  => mean_gap={np.mean(gaps):+.4f} median={np.median(gaps):+.4f} "
                  f"n_cells={len(gaps)} sign_neg={sum(1 for g in gaps if g<0)}/{len(gaps)}")

    # Also compare to v2 baseline on math500
    print("\n=== v2 baseline MATH-500 stratified gap (p_contam thresholds) ===")
    v2_gaps = []
    for i, m in enumerate(models):
        correct = load_math500_correct(m)
        if correct is None:
            continue
        p_v2 = load_v2_pcontam(m)
        if p_v2 is None:
            continue
        # Flip: for v2, HIGH p_contam = contaminated, LOW = clean.
        # Convert to peer_z-convention: detector_score where high = clean, low = contam.
        # Use -p_v2 as a score. Then contam = -p_v2 <= -0.5 (i.e. p >= 0.5);
        # clean = -p_v2 >= -0.1 (i.e. p <= 0.1). Use stratified_gap accordingly.
        det = -p_v2
        stat = stratified_gap(correct, det, low_thr=-0.5, high_thr=-0.1)
        if stat is None:
            continue
        v2_gaps.append(stat["gap"])
        sign = "CORRECT" if stat["gap"] < 0 else "wrong  "
        print(f"  {m:<45s} n={stat['n_clean']:>3d}/{stat['n_contam']:<3d} "
               f"clean={stat['clean_acc']:.3f} contam={stat['contam_acc']:.3f} "
               f"gap={stat['gap']:+.3f} [{sign}]")
    if v2_gaps:
        print(f"  => mean_gap={np.mean(v2_gaps):+.4f} median={np.median(v2_gaps):+.4f} "
              f"n_cells={len(v2_gaps)} sign_neg={sum(1 for g in v2_gaps if g<0)}/{len(v2_gaps)}")


if __name__ == "__main__":
    main()
