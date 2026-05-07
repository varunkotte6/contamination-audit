#!/usr/bin/env python3
"""v6 analysis v3: Mann-Whitney U on peer_z split by correctness.

For each model m, compute the Mann-Whitney U statistic between
  peer_z_dd(m, i) on items where correct(m, i) = 1,
and
  peer_z_dd(m, i) on items where correct(m, i) = 0.

Under a working detector:
- correct items are more memorized-looking (lower peer_z)
- so U should be small (U tests if distribution-1 < distribution-2)
- we report one-sided p-value for H1: peer_z(correct) < peer_z(incorrect)

Comparison: v2 baseline does the same with p_contam (high p_contam should
align with correct items if v2 is working; one-sided p for p_v2(correct) >
p_v2(incorrect)).

The test is robust to threshold choice because it uses the full
distribution. It also doesn't need cell-count filters since we operate
within-model on all items.
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

SPECIALISTS = {"Qwen__Qwen2.5-Math-7B", "Qwen__Qwen2.5-Coder-7B"}


def load_all_answer_losses():
    files = sorted(os.listdir(ANS_DIR))
    models = []; mats = []
    for f in files:
        if not f.endswith(".npz"):
            continue
        slug = f[:-4]
        d = np.load(f"{ANS_DIR}/{f}")
        models.append(slug); mats.append(d["answer_loss"])
    if not mats:
        return [], np.zeros((0, 500))
    n = max(len(m) for m in mats)
    L = np.full((len(mats), n), np.nan, dtype=np.float32)
    for i, m in enumerate(mats):
        L[i, :len(m)] = m
    return models, L


def double_standardized_z(L, peer_mask=None):
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
    peer_mask = np.array([m not in SPECIALISTS for m in models])
    pz = double_standardized_z(L, peer_mask=peer_mask)
    print(f"\n{'model':<45s} "
          f"{'n_corr':>6s} {'n_inc':>5s} "
          f"{'v6_pz_c':>8s} {'v6_pz_w':>8s} {'v6_U_p':>8s} "
          f"{'v2_pc_c':>8s} {'v2_pc_w':>8s} {'v2_U_p':>8s}")
    v6_ps = []; v2_ps = []
    v6_diffs = []; v2_diffs = []
    for i, m in enumerate(models):
        correct = load_math500_correct(m)
        if correct is None:
            continue
        n = min(len(correct), pz.shape[1])
        correct = correct[:n]
        v6 = pz[i, :n]
        # Mann-Whitney U: H1 v6(correct) < v6(incorrect), i.e. correct items are memorized
        n_c = int(correct.sum()); n_i = int((1 - correct).sum())
        if n_c < 10 or n_i < 10:
            continue
        v6c = v6[correct == 1]; v6w = v6[correct == 0]
        v6c_valid = v6c[~np.isnan(v6c)]; v6w_valid = v6w[~np.isnan(v6w)]
        if len(v6c_valid) < 10 or len(v6w_valid) < 10:
            continue
        u_v6, p_v6 = stats.mannwhitneyu(v6c_valid, v6w_valid, alternative="less")
        mean_v6c = float(np.mean(v6c_valid)); mean_v6w = float(np.mean(v6w_valid))
        v6_diffs.append(mean_v6c - mean_v6w)
        v6_ps.append(p_v6)
        # v2
        p_v2 = load_v2_pcontam(m)
        if p_v2 is not None:
            p_v2 = p_v2[:n]
            v2c = p_v2[correct == 1]; v2w = p_v2[correct == 0]
            v2c_valid = v2c[~np.isnan(v2c)]; v2w_valid = v2w[~np.isnan(v2w)]
            u_v2, p_v2stat = stats.mannwhitneyu(v2c_valid, v2w_valid, alternative="greater")
            mean_v2c = float(np.mean(v2c_valid)); mean_v2w = float(np.mean(v2w_valid))
            v2_diffs.append(mean_v2c - mean_v2w)
            v2_ps.append(p_v2stat)
        else:
            mean_v2c = np.nan; mean_v2w = np.nan; p_v2stat = np.nan
        print(f"{m:<45s} "
               f"{n_c:>6d} {n_i:>5d} "
               f"{mean_v6c:>+.3f} {mean_v6w:>+.3f} {p_v6:>8.1e} "
               f"{mean_v2c:>+.3f} {mean_v2w:>+.3f} {p_v2stat:>8.1e}")
    print()
    if v6_ps:
        n_sig = sum(1 for p in v6_ps if p < 0.05)
        n_corr_dir = sum(1 for d in v6_diffs if d < 0)
        print(f"v6: {n_sig}/{len(v6_ps)} significant (MW one-sided p<0.05, correct-direction), "
              f"{n_corr_dir}/{len(v6_diffs)} correct-direction mean diff")
    if v2_ps:
        n_sig = sum(1 for p in v2_ps if p < 0.05)
        n_corr_dir = sum(1 for d in v2_diffs if d > 0)
        print(f"v2: {n_sig}/{len(v2_ps)} significant (MW one-sided p<0.05, correct-direction), "
              f"{n_corr_dir}/{len(v2_diffs)} correct-direction mean diff")


if __name__ == "__main__":
    main()
