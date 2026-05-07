#!/usr/bin/env python3
"""v4: stratify deployment gap by an independent-family difficulty proxy.

v3 used Pythia-1B loss (same family as the reference) to difficulty-match
calibration negatives and collapsed the uniform-difficulty Bonferroni count
from 4 cells to 1. The theory section argues a working fix needs a
difficulty estimator independent of the reference-model family. v4 tests
this at the DEPLOYMENT analysis layer: for each audit cell, we stratify
items by OLMo-2-7B per-item loss (cross-family) quartile and compute the
within-stratum clean-minus-contaminated gap using the existing v2 flags,
combining strata by inverse-variance weighting.

If v4's iv-weighted gap is less positive (less wrong-direction) than v2's
unstratified gap on variant-difficulty benchmarks, the fix works.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = "{REPO_ROOT}"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5 = f"{PROJECT}/results/phase5_full"
OUT = f"{PROJECT}/results/v4_olmo_strat"
Path(OUT).mkdir(parents=True, exist_ok=True)

REFLOSS_BASE = f"{PROJECT}/results/refmodel_sensitivity/refloss_olmo27b.npz"
REFLOSS_EXT = f"{PROJECT}/results/refmodel_sensitivity/refloss_olmo27b_extended.npz"

MATH500_BENCH = "math500"
LIVECB_BENCH = "livecodebench"


def load_refloss():
    out = {}
    if os.path.exists(REFLOSS_BASE):
        d = np.load(REFLOSS_BASE)
        for k in d.files:
            out[k] = d[k]
    if os.path.exists(REFLOSS_EXT):
        d = np.load(REFLOSS_EXT)
        for k in d.files:
            out[k] = d[k]
    return out


def load_correct(model, bench):
    # Variant-difficulty benchmarks live in separate directories
    if bench == "math500":
        p = f"{PROJECT}/results/math_eval/{model}_math500/results.pkl"
        if os.path.exists(p):
            r = pickle.load(open(p, "rb"))
            return np.array([x["correct"] for x in r], dtype=float)
        return None
    if bench == "livecodebench":
        p = f"{PROJECT}/results/livecodebench_sandbox/{model}/results.pkl"
        if os.path.exists(p):
            r = pickle.load(open(p, "rb"))
            if isinstance(r, list):
                return np.array([x.get("correct", x.get("passed", 0)) for x in r],
                                 dtype=float)
        return None
    path = f"{PHASE5}/{model}/per_item.pkl"
    if not os.path.exists(path):
        return None
    d = pickle.load(open(path, "rb"))
    if bench not in d:
        return None
    items = d[bench]["per_item"]
    return np.array([it["correct"] for it in items], dtype=float)


def load_v2(model, bench):
    sp = f"{PHASE4_V2}/{model}/scores.pkl"
    if not os.path.exists(sp):
        return None
    d = pickle.load(open(sp, "rb"))
    idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench]
    if not idxs:
        return None
    item_ids = np.array([d["provenance"][i][1] for i in idxs])
    order = np.argsort(item_ids)
    p_contam = d["p_contam"][idxs][order]
    item_ids = item_ids[order]
    return p_contam, item_ids


def iv_weighted_strat_gap(correct, p_contam, difficulty, n_strata=4,
                           min_per_stratum=5):
    """Stratify by difficulty quantiles; compute within-stratum
    clean-contam gap with inverse-variance weighting."""
    n = min(len(correct), len(p_contam), len(difficulty))
    correct = correct[:n]; p_contam = p_contam[:n]; difficulty = difficulty[:n]
    # Drop NaN-difficulty items
    valid = ~np.isnan(difficulty)
    correct = correct[valid]; p_contam = p_contam[valid]
    difficulty = difficulty[valid]
    if len(difficulty) < n_strata * min_per_stratum:
        return None
    q_edges = np.quantile(difficulty, np.linspace(0, 1, n_strata + 1))
    q_edges[-1] += 1e-9
    strat = np.digitize(difficulty, q_edges[1:-1])
    gaps = []
    weights = []
    for s in range(n_strata):
        mask = strat == s
        if mask.sum() < min_per_stratum * 2:
            continue
        p = p_contam[mask]; c = correct[mask]
        clean = p < 0.1; contam = p > 0.5
        if clean.sum() < min_per_stratum or contam.sum() < min_per_stratum:
            continue
        cc = c[clean]; ct = c[contam]
        gap = cc.mean() - ct.mean()
        # variance of mean-diff
        var = cc.var(ddof=1) / len(cc) + ct.var(ddof=1) / len(ct)
        if var <= 0:
            continue
        gaps.append(gap)
        weights.append(1 / var)
    if not gaps:
        return None
    gaps = np.array(gaps); weights = np.array(weights)
    iv_gap = np.sum(gaps * weights) / np.sum(weights)
    iv_se = np.sqrt(1 / np.sum(weights))
    z = iv_gap / iv_se
    p_two = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(n_strata=len(gaps), iv_gap=float(iv_gap),
                iv_se=float(iv_se), z=float(z), p_two=float(p_two))


def unstrat_gap(correct, p_contam, min_n=10):
    n = min(len(correct), len(p_contam))
    correct = correct[:n]; p_contam = p_contam[:n]
    clean = p_contam < 0.1; contam = p_contam > 0.5
    if clean.sum() < min_n or contam.sum() < min_n:
        return None
    cc = correct[clean]; ct = correct[contam]
    return dict(obs_gap=float(cc.mean() - ct.mean()),
                n_clean=int(clean.sum()), n_contam=int(contam.sum()))


def main():
    refloss = load_refloss()
    print(f"OLMo refloss benchmarks cached: {list(refloss.keys())}")
    models = sorted(os.listdir(PHASE4_V2))

    rows = []
    for bench in refloss.keys():
        olmo_diff = refloss[bench]
        for model in models:
            if model.startswith("_"):
                continue
            correct = load_correct(model, bench)
            if correct is None:
                continue
            v2 = load_v2(model, bench)
            if v2 is None:
                continue
            p_contam, item_ids = v2
            # Align on first N items
            n = min(len(correct), len(olmo_diff), len(p_contam))
            if n < 30:
                continue
            u = unstrat_gap(correct[:n], p_contam[:n])
            iv = iv_weighted_strat_gap(correct[:n], p_contam[:n],
                                        olmo_diff[:n])
            if u is None:
                continue
            row = {"model": model, "benchmark": bench,
                    "unstrat_gap": u["obs_gap"],
                    "n_clean": u["n_clean"], "n_contam": u["n_contam"]}
            if iv is not None:
                row.update({k: iv[k] for k in ["iv_gap", "iv_se", "p_two",
                                                "n_strata"]})
            rows.append(row)

    import csv
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(f"{OUT}/v4_per_cell.csv", "w") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {len(rows)} cells to v4_per_cell.csv")

    # Aggregate per benchmark
    summary = {}
    for bench in refloss.keys():
        sub = [r for r in rows if r["benchmark"] == bench]
        if not sub:
            continue
        unstrat = np.array([r["unstrat_gap"] for r in sub])
        ivs = np.array([r.get("iv_gap", np.nan) for r in sub])
        valid_iv = ~np.isnan(ivs)
        summary[bench] = {
            "n_cells": len(sub),
            "v2_unstratified_mean_gap": float(unstrat.mean()),
            "v4_olmo_stratified_mean_gap": float(np.nanmean(ivs)) if valid_iv.any() else None,
            "n_cells_with_strat": int(valid_iv.sum()),
            "improvement_pp": (float((unstrat[valid_iv] - ivs[valid_iv]).mean())
                                 if valid_iv.any() else None),
        }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
