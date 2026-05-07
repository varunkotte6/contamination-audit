#!/usr/bin/env python3
"""v6 analysis: rank-correlation and per-cell structure.

We look at two metrics:

1. Spearman correlation between peer_z_dd(m, i) and correct(m, i) across
   items for each model m. Under v6, if the detector is working, LOW
   peer_z (anomalously easy for m) should correspond to HIGH accuracy
   (the model 'knows' the answer), giving a NEGATIVE correlation
   between peer_z and correct. Under v2 / random detectors, the
   correlation is near zero or positive.

2. Per-cell stratified gap using continuous thresholds.

We also compute these for peer_z computed on a restricted "general-purpose"
peer pool (excluding known math/code specialists) to test whether
excluding specialists sharpens the signal.
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

# Domain specialists that bias the math-peer median when included.
SPECIALISTS = {"Qwen__Qwen2.5-Math-7B", "Qwen__Qwen2.5-Coder-7B"}


def load_all_answer_losses():
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
    n = max(len(m) for m in mats)
    L = np.full((len(mats), n), np.nan, dtype=np.float32)
    for i, m in enumerate(mats):
        L[i, :len(m)] = m
    return models, L


def double_standardized_z(L, peer_mask=None):
    """Two-stage normalization. peer_mask (bool array over models) restricts
    the peer pool used for the cross-model median; within-model
    standardization uses the full item range."""
    mean_within = np.nanmean(L, axis=1, keepdims=True)
    std_within = np.nanstd(L, axis=1, keepdims=True) + 1e-6
    z_within = (L - mean_within) / std_within
    peer_L = z_within if peer_mask is None else z_within[peer_mask]
    med = np.nanmedian(peer_L, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(peer_L - med), axis=0, keepdims=True)
    scale = 1.4826 * mad + 1e-6
    return (z_within - med) / scale


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


def main():
    models, L = load_all_answer_losses()
    print(f"Loaded {len(models)} models on {L.shape[1]} items")
    for peer_label, peer_mask in [
        ("ALL", np.ones(len(models), dtype=bool)),
        ("NO-SPECIALISTS",
          np.array([m not in SPECIALISTS for m in models])),
    ]:
        print(f"\n========== peer pool: {peer_label} "
              f"({peer_mask.sum()} / {len(models)} models) ==========")
        pz = double_standardized_z(L, peer_mask=peer_mask)
        # Rank-correlation: peer_z vs correct. We want NEGATIVE correlation
        # (low peer_z = memorized = correct; high peer_z = not memorized
        # = incorrect) for the detector to be working.
        print(f"{'model':<45s} {'n':>4s} {'rho':>6s} {'p_rho':>8s} "
               f"{'v2_rho':>7s} {'v2_p':>8s}")
        v6_rhos, v2_rhos = [], []
        for i, m in enumerate(models):
            correct = load_math500_correct(m)
            if correct is None:
                continue
            n = min(len(correct), pz.shape[1])
            correct = correct[:n]
            det_v6 = pz[i, :n]
            valid = ~np.isnan(det_v6)
            if valid.sum() < 30:
                continue
            # Spearman correlation with the detector
            # A working detector: low score = memorized = correct => negative rho
            rho_v6, p_v6 = stats.spearmanr(det_v6[valid], correct[valid])
            # v2 baseline: high p_contam = contaminated = correct if
            # memorization helps => negative rho means v2 is ANTI-working.
            # For comparison, we use -p_v2 so that "high detector score = clean"
            # like v6. Then both detectors want negative rho.
            p_v2 = load_v2_pcontam(m)
            if p_v2 is None:
                rho_v2 = np.nan; p_v2s = np.nan
            else:
                p_v2 = p_v2[:n]
                valid_v2 = ~np.isnan(p_v2)
                det_v2 = -p_v2[valid_v2]
                c_v2 = correct[valid_v2]
                rho_v2, p_v2s = stats.spearmanr(det_v2, c_v2)
            print(f"{m:<45s} {valid.sum():>4d} {rho_v6:>+.3f} "
                   f"{p_v6:>8.1e} {rho_v2:>+.3f} {p_v2s:>8.1e}")
            v6_rhos.append(rho_v6)
            if not np.isnan(rho_v2):
                v2_rhos.append(rho_v2)
        if v6_rhos:
            print(f"  v6 mean_rho = {np.mean(v6_rhos):+.4f}, "
                  f"n_neg = {sum(1 for x in v6_rhos if x < 0)}/{len(v6_rhos)}")
        if v2_rhos:
            print(f"  v2 mean_rho = {np.mean(v2_rhos):+.4f}, "
                  f"n_neg = {sum(1 for x in v2_rhos if x < 0)}/{len(v2_rhos)}")


if __name__ == "__main__":
    main()
