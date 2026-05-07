#!/usr/bin/env python3
"""W3: null-peer control for v8.

The reviewer asks whether v8's signal is driven by genuine
capability-matched peers or by any comparison against arbitrary peers.
We shuffle the peer assignments 100 times — for each audited model m
we pick K peers UNIFORMLY AT RANDOM from the audit matrix, excluding
m itself and same-family peers — compute the v8 MATH-500 stratified
gap per shuffle, and report the distribution.

For each model and each shuffle, we use:
- within-model z of answer_loss (same as real v8)
- cross-model delta vs the K random peers' mean z
- 20/80 quantile thresholds
- stratified gap = clean_acc - contam_acc

We then compare the real v8 gap to the shuffled distribution per cell
(two-sided) and aggregate.
"""
import os
import pickle
import json
import numpy as np

PROJECT = "{REPO_ROOT}"
ANS_DIR = f"{PROJECT}/results/v6_answer_loss"
MATH_EVAL = f"{PROJECT}/results/math_eval"
OUT = f"{PROJECT}/results/v8_null_control"
os.makedirs(OUT, exist_ok=True)

K = 7
TAIL = 0.2
N_SHUFFLES = 100
SEED = 0


def fam(slug):
    s = slug.lower()
    for k in ["pythia", "qwen", "gemma", "olmo", "llama", "phi", "mistral",
               "yi-", "deepseek", "granite"]:
        if k in s:
            return k
    return "other"


def within_z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6)


def stratified_gap(correct, delta, q=TAIL, min_n=10):
    valid = ~np.isnan(delta)
    correct = correct[valid]; delta = delta[valid]
    if len(delta) < 30:
        return None
    lo = np.quantile(delta, q); hi = np.quantile(delta, 1 - q)
    contam = delta <= lo; clean = delta >= hi
    if contam.sum() < min_n or clean.sum() < min_n:
        return None
    return float(correct[clean].mean() - correct[contam].mean())


def main():
    slugs = sorted([f[:-4] for f in os.listdir(ANS_DIR) if f.endswith(".npz")])
    al_raw = {s: np.load(f"{ANS_DIR}/{s}.npz")["answer_loss"] for s in slugs}
    al = {s: within_z(al_raw[s]) for s in slugs}
    accs = {}
    for s in slugs:
        p = f"{MATH_EVAL}/{s}_math500/results.pkl"
        if os.path.exists(p):
            r = pickle.load(open(p, "rb"))
            accs[s] = float(np.mean([x["correct"] for x in r]))
        else:
            accs[s] = None

    rng = np.random.default_rng(SEED)
    results = {}
    for m in slugs:
        if accs.get(m) is None:
            continue
        m_al = al[m]
        m_correct = np.array([x["correct"]
                               for x in pickle.load(open(f"{MATH_EVAL}/{m}_math500/results.pkl", "rb"))],
                              dtype=float)
        n = min(len(m_al), len(m_correct))
        m_al_n = m_al[:n]; m_correct_n = m_correct[:n]
        m_fam = fam(m)

        # Real v8: capability-matched peers (±0.15 acc, exclude same-family)
        real_peers = [s for s in slugs
                      if s != m and accs.get(s) is not None
                      and abs(accs[s] - accs[m]) <= 0.15
                      and fam(s) != m_fam]
        if len(real_peers) < 3:
            continue
        real_peers = sorted(real_peers, key=lambda s: abs(accs[s] - accs[m]))[:K]
        real_peer_al = np.stack([al[s][:n] for s in real_peers], axis=0)
        valid = ~np.isnan(m_al_n) & ~np.isnan(real_peer_al).any(axis=0)
        real_delta = m_al_n[valid] - real_peer_al[:, valid].mean(axis=0)
        real_gap = stratified_gap(m_correct_n[valid], real_delta)
        if real_gap is None:
            continue

        # Null: shuffled peers (random K peers, excluding self and same family)
        candidates = [s for s in slugs if s != m and fam(s) != m_fam
                      and accs.get(s) is not None]
        if len(candidates) < K:
            continue
        shuf_gaps = []
        for i in range(N_SHUFFLES):
            peer_idx = rng.choice(len(candidates), size=K, replace=False)
            shuf_peers = [candidates[j] for j in peer_idx]
            shuf_peer_al = np.stack([al[s][:n] for s in shuf_peers], axis=0)
            v2 = ~np.isnan(m_al_n) & ~np.isnan(shuf_peer_al).any(axis=0)
            d = m_al_n[v2] - shuf_peer_al[:, v2].mean(axis=0)
            g = stratified_gap(m_correct_n[v2], d)
            if g is not None:
                shuf_gaps.append(g)
        shuf_gaps = np.array(shuf_gaps)
        # Compute: what fraction of shuffled gaps are <= real_gap
        # (lower gap = more correct-direction; we want real to be in the left tail)
        if len(shuf_gaps):
            frac_lower = float((shuf_gaps <= real_gap).mean())
        else:
            frac_lower = None
        results[m] = {
            "acc": accs[m],
            "real_gap": real_gap,
            "null_gap_mean": float(shuf_gaps.mean()) if len(shuf_gaps) else None,
            "null_gap_std": float(shuf_gaps.std()) if len(shuf_gaps) else None,
            "null_gap_5pct": float(np.percentile(shuf_gaps, 5)) if len(shuf_gaps) else None,
            "frac_null_le_real": frac_lower,
            "n_shuffles": len(shuf_gaps),
        }

    print(f"{'model':<45s} {'acc':>5s} {'real':>7s} {'null_mean':>9s} "
          f"{'null_std':>9s} {'null_5%':>8s} {'frac_null<=real':>15s}")
    for m, r in sorted(results.items(), key=lambda kv: kv[1]["acc"]):
        ng_mean = f"{r['null_gap_mean']:+.3f}" if r["null_gap_mean"] is not None else "   n/a"
        ng_std = f"{r['null_gap_std']:.3f}" if r["null_gap_std"] is not None else "  n/a"
        ng5 = f"{r['null_gap_5pct']:+.3f}" if r["null_gap_5pct"] is not None else "   n/a"
        fnr = f"{r['frac_null_le_real']:.3f}" if r["frac_null_le_real"] is not None else "  n/a"
        print(f"{m:<45s} {r['acc']:>.3f} {r['real_gap']:>+.3f} {ng_mean:>9s} "
              f"{ng_std:>9s} {ng5:>8s} {fnr:>15s}")

    # Aggregate: count cells where real_gap is in bottom 5% of null distribution
    cells_vs_null_5 = sum(1 for r in results.values()
                           if r["frac_null_le_real"] is not None
                           and r["frac_null_le_real"] < 0.05)
    cells_vs_null_1 = sum(1 for r in results.values()
                           if r["frac_null_le_real"] is not None
                           and r["frac_null_le_real"] < 0.01)
    print(f"\nCells where real v8 gap is in bottom 5% of null: "
          f"{cells_vs_null_5}/{len(results)}")
    print(f"Cells where real v8 gap is in bottom 1% of null: "
          f"{cells_vs_null_1}/{len(results)}")
    # Mean real vs mean null
    real_gaps = [r["real_gap"] for r in results.values()]
    null_means = [r["null_gap_mean"] for r in results.values()
                   if r["null_gap_mean"] is not None]
    print(f"Mean real v8 gap: {np.mean(real_gaps):+.4f}")
    print(f"Mean null v8 gap: {np.mean(null_means):+.4f}")
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
