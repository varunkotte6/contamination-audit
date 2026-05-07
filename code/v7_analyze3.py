#!/usr/bin/env python3
"""v7 analysis v3: filter-then-split on orig_loss and delta.

Among items where the model is CONFIDENT (low orig_loss), split by
delta:
- confident + low delta = MEMORIZED (prefers answer regardless of problem)
- confident + high delta = DERIVED (prefers answer tied to this problem)

The 'confident' filter is critical because it removes clueless items
(high orig_loss, regardless of delta) that would otherwise dominate
the low-delta bucket.
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


def main():
    slugs = sorted([f[:-4] for f in os.listdir(V7_DIR) if f.endswith(".npz")])
    print(f"v7 slugs: {slugs}")
    for conf_q in [0.3, 0.5]:
        for split_q in [0.25, 0.33]:
            print(f"\n=== conf=lowest {int(conf_q*100)}% of orig_loss, "
                  f"split=quartiles ({split_q}, {1-split_q}) of delta ===")
            print(f"{'model':<45s} {'n_conf':>6s} {'n_cl':>4s} {'n_ct':>4s} "
                  f"{'cl_acc':>6s} {'ct_acc':>6s} {'gap':>7s}")
            gaps = []
            for slug in slugs:
                v7 = load_v7(slug)
                if v7 is None:
                    continue
                delta = v7["delta"]; orig = v7["orig_loss"]
                correct = load_math500_correct(slug)
                if correct is None:
                    continue
                n = min(len(delta), len(correct))
                delta = delta[:n]; orig = orig[:n]; correct = correct[:n]
                valid = ~np.isnan(delta) & ~np.isnan(orig)
                delta = delta[valid]; orig = orig[valid]; correct = correct[valid]
                if len(orig) < 50:
                    continue
                # Filter to confident items
                conf_thr = np.quantile(orig, conf_q)
                conf_mask = orig <= conf_thr
                if conf_mask.sum() < 30:
                    continue
                d_conf = delta[conf_mask]
                c_conf = correct[conf_mask]
                # Split by delta quartiles within confident items
                lo = np.quantile(d_conf, split_q)
                hi = np.quantile(d_conf, 1 - split_q)
                contam = d_conf <= lo
                clean = d_conf >= hi
                if contam.sum() < 10 or clean.sum() < 10:
                    continue
                gap = float(c_conf[clean].mean() - c_conf[contam].mean())
                gaps.append(gap)
                print(f"{slug:<45s} {int(conf_mask.sum()):>6d} "
                       f"{int(clean.sum()):>4d} {int(contam.sum()):>4d} "
                       f"{c_conf[clean].mean():>+.3f} {c_conf[contam].mean():>+.3f} "
                       f"{gap:>+.3f}")
            if gaps:
                n_neg = sum(1 for g in gaps if g < 0)
                print(f"v7 mean_gap={np.mean(gaps):+.4f} median={np.median(gaps):+.4f} "
                      f"sign_neg={n_neg}/{len(gaps)}")


if __name__ == "__main__":
    main()
