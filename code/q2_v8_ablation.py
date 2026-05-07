#!/usr/bin/env python3
"""Q2: v8 sensitivity ablation.

Reviewer asks: what happens to the 6 Bonferroni-surviving MATH-500
cells under (a) K ∈ {3, 5, 10}, (b) accuracy windows ±0.10 / ±0.20,
(c) random peer assignment (null), (d) removal of same-sub-family
peers within Qwen (Coder/Math count as same family).
"""
import os
import pickle
import numpy as np

PROJECT = "{REPO_ROOT}"
ANS_DIR = f"{PROJECT}/results/v6_answer_loss"
MATH_EVAL = f"{PROJECT}/results/math_eval"


def fam_basic(slug):
    s = slug.lower()
    for k in ["pythia", "qwen", "gemma", "olmo", "llama", "phi", "mistral",
               "yi-", "deepseek", "granite"]:
        if k in s:
            return k
    return "other"


def fam_strict(slug):
    """Treat all Qwen-2.5 sub-variants (Coder, Math, base) as the same
    sub-family so that their losses cannot anchor each other."""
    s = slug.lower()
    if "qwen" in s:
        if "qwen2.5" in s or "qwen2-5" in s or "qwen-2.5" in s:
            return "qwen2.5"
        if "qwen3" in s or "qwen-3" in s:
            return "qwen3"
        return "qwen"
    return fam_basic(slug)


def within_z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6)


def perm_p(cl, ct, n_perm=2000, seed=42):
    rng = np.random.default_rng(seed)
    obs = cl.mean() - ct.mean()
    comb = np.concatenate([cl, ct])
    nle = 0
    for _ in range(n_perm):
        rng.shuffle(comb)
        if comb[:len(cl)].mean() - comb[len(cl):].mean() <= obs:
            nle += 1
    return nle / n_perm


def evaluate(slugs, al, accs, all_correct, fam_fn, K, max_diff,
              tail=0.2, random_peers=False, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in slugs:
        if accs.get(m) is None:
            continue
        m_acc = accs[m]
        m_fam = fam_fn(m)
        if random_peers:
            cands = [s for s in slugs if s != m and fam_fn(s) != m_fam
                     and accs.get(s) is not None]
            if len(cands) < K:
                continue
            ref_slugs = list(rng.choice(cands, K, replace=False))
        else:
            others = [(s, abs(accs[s] - m_acc)) for s in slugs
                      if s != m and accs.get(s) is not None
                      and abs(accs[s] - m_acc) <= max_diff
                      and fam_fn(s) != m_fam]
            if len(others) < 3:
                continue
            others.sort(key=lambda kv: kv[1])
            ref_slugs = [s for s, _ in others[:K]]
        ref_al = np.stack([al[s] for s in ref_slugs], axis=0)
        m_al = al[m]
        c = all_correct.get(m)
        if c is None:
            continue
        n = min(len(m_al), ref_al.shape[1], len(c))
        m_n = m_al[:n]; ref_n = ref_al[:, :n]; c_n = c[:n]
        valid = ~np.isnan(m_n) & ~np.isnan(ref_n).any(axis=0)
        delta = m_n[valid] - ref_n[:, valid].mean(axis=0)
        cv = c_n[valid]
        lo = np.quantile(delta, tail); hi = np.quantile(delta, 1 - tail)
        contam = delta <= lo; clean = delta >= hi
        if contam.sum() < 10 or clean.sum() < 10:
            continue
        gap = float(cv[clean].mean() - cv[contam].mean())
        p = perm_p(cv[clean], cv[contam])
        rows.append({"model": m, "gap": gap, "perm_p": p,
                     "n_peers": len(ref_slugs)})
    if not rows:
        return None
    gaps = [r["gap"] for r in rows]
    ps = [r["perm_p"] for r in rows]
    n_neg = sum(1 for g in gaps if g < 0)
    n_sig = sum(1 for p_ in ps if p_ < 0.05)
    n_bonf = sum(1 for p_ in ps if p_ < 0.05 / len(rows))
    return {
        "n_cells": len(rows), "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "n_neg": n_neg, "n_sig": n_sig, "n_bonf": n_bonf,
    }


def main():
    slugs = sorted([f[:-4] for f in os.listdir(ANS_DIR) if f.endswith(".npz")])
    al_raw = {s: np.load(f"{ANS_DIR}/{s}.npz")["answer_loss"] for s in slugs}
    al = {s: within_z(al_raw[s]) for s in slugs}
    accs = {}
    all_correct = {}
    for s in slugs:
        p = f"{MATH_EVAL}/{s}_math500/results.pkl"
        if os.path.exists(p):
            r = pickle.load(open(p, "rb"))
            all_correct[s] = np.array([x["correct"] for x in r], dtype=float)
            accs[s] = float(np.mean(all_correct[s]))
        else:
            accs[s] = None

    print("v8 sensitivity ablation on MATH-500 (within-model standardized)\n")
    print(f"{'config':<55s} {'n_cells':>7s} {'mean_gap':>9s} {'n_neg':>6s} "
          f"{'n_sig':>6s} {'n_bonf':>7s}")

    # Baseline: K=7, max_diff=0.15, basic family
    for K in [3, 5, 7, 10]:
        r = evaluate(slugs, al, accs, all_correct, fam_basic, K, 0.15)
        if r:
            print(f"K={K:<2d} window=±0.15 fam=basic{'':<28s} "
                   f"{r['n_cells']:>7d} {r['mean_gap']:>+.4f} {r['n_neg']:>6d} "
                   f"{r['n_sig']:>6d} {r['n_bonf']:>7d}")

    print()
    for w in [0.10, 0.15, 0.20]:
        r = evaluate(slugs, al, accs, all_correct, fam_basic, 7, w)
        if r:
            print(f"K=7 window=±{w:.2f} fam=basic{'':<28s} "
                   f"{r['n_cells']:>7d} {r['mean_gap']:>+.4f} {r['n_neg']:>6d} "
                   f"{r['n_sig']:>6d} {r['n_bonf']:>7d}")

    print()
    # Strict family (Qwen2.5 sub-family fused)
    for K in [5, 7]:
        r = evaluate(slugs, al, accs, all_correct, fam_strict, K, 0.15)
        if r:
            print(f"K={K:<2d} window=±0.15 fam=strict (Qwen2.5 fused){'':<13s} "
                   f"{r['n_cells']:>7d} {r['mean_gap']:>+.4f} {r['n_neg']:>6d} "
                   f"{r['n_sig']:>6d} {r['n_bonf']:>7d}")

    print()
    # Random-peer null (averaged over 100 shuffles)
    print("Random-peer null (100 shuffles, K=7):")
    nulls = []
    for seed in range(100):
        r = evaluate(slugs, al, accs, all_correct, fam_basic, 7, 0.15,
                      random_peers=True, seed=seed)
        if r:
            nulls.append(r)
    null_mean_gap = np.mean([r["mean_gap"] for r in nulls])
    null_n_sig = np.mean([r["n_sig"] for r in nulls])
    null_n_bonf = np.mean([r["n_bonf"] for r in nulls])
    print(f"  Random-peer null: mean_gap={null_mean_gap:+.4f}, "
          f"avg n_sig={null_n_sig:.2f}, avg n_bonf={null_n_bonf:.2f}")


if __name__ == "__main__":
    main()
