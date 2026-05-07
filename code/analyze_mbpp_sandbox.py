#!/usr/bin/env python3
"""Contamination analysis of sandboxed MBPP pass@1 scores.

For each of the 11 audit models, load the sandboxed pass@1 results and
slice by p_contam (v1 and v2) from phase4. Compute per-cell
clean-vs-contaminated accuracy gap, permutation p-value, and aggregate
across models.

Output goes to results/stat_tests/stat_tests_v2_with_mbpp.csv for
downstream stratified analysis / reviewer-response tables.
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT = "{REPO_ROOT}"
MBPP_SANDBOX = f"{PROJECT}/results/mbpp_sandbox"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
OUT = f"{PROJECT}/results/stat_tests"


def load_p_contam(phase4_dir, model, bench):
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


def perm_test(cc, ct, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = cc.mean() - ct.mean()
    combined = np.concatenate([cc, ct])
    extreme = 0
    n_cc = len(cc)
    for _ in range(n_perm):
        rng.shuffle(combined)
        c = combined[:n_cc].mean() - combined[n_cc:].mean()
        if c <= obs:
            extreme += 1
    return obs, extreme / n_perm


def bootstrap_ci(cc, ct, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(cc, size=len(cc), replace=True).mean()
        b = rng.choice(ct, size=len(ct), replace=True).mean()
        gaps[i] = a - b
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def cell_stat(correct, p, min_n=10):
    clean = p < 0.1
    contam = p > 0.5
    nc = int(clean.sum()); nt = int(contam.sum())
    if nc < min_n or nt < min_n:
        return None
    cc = correct[clean]; ct = correct[contam]
    gap, perm_p = perm_test(cc, ct)
    ci_lo, ci_hi = bootstrap_ci(cc, ct)
    return {"n_clean": nc, "n_contam": nt,
            "clean_acc": float(cc.mean()), "contam_acc": float(ct.mean()),
            "obs_gap": float(gap), "ci_low": ci_lo, "ci_high": ci_hi,
            "perm_p": float(perm_p)}


def main():
    rows = []
    for m in sorted(os.listdir(MBPP_SANDBOX)):
        if m.startswith("_"):
            continue
        rf = f"{MBPP_SANDBOX}/{m}/results.pkl"
        if not os.path.exists(rf):
            continue
        with open(rf, "rb") as f:
            results = pickle.load(f)
        correct = np.array([int(r["passed"]) for r in results], dtype=np.int32)
        for version, ph4 in [("v1", PHASE4_V1), ("v2", PHASE4_V2)]:
            p = load_p_contam(ph4, m, "mbpp")
            if p is None:
                continue
            N = min(len(correct), len(p))
            stat = cell_stat(correct[:N], p[:N])
            if stat is None:
                print(f"  [skip] {m} v{version}: n_clean={np.sum(p[:N]<0.1)} n_contam={np.sum(p[:N]>0.5)} below floor")
                continue
            rows.append({
                "model": m.replace("__", "/"),
                "benchmark": "mbpp_sandbox", "ensemble": version,
                "evaluator": "sandbox_pass1",
                **stat,
            })

    df = pd.DataFrame(rows)
    print(f"\nTotal mbpp_sandbox cells: {len(df)}")
    for ens in ["v1", "v2"]:
        sub = df[df.ensemble == ens]
        if len(sub) == 0: continue
        n_correct = ((sub.obs_gap < 0) & (sub.perm_p < 0.05)).sum()
        n_wrong = ((sub.obs_gap > 0) & (sub.perm_p > 0.95)).sum()
        print(f"  {ens}: n={len(sub)} mean_gap={sub.obs_gap.mean():+.4f} "
              f"sig_correct={n_correct} sig_wrong={n_wrong}")
        # BH-FDR
        corr_mask = sub.obs_gap < 0
        if corr_mask.any():
            pvals = [min(p, 1-p)*2 for p in sub.loc[corr_mask, "perm_p"]]
            rej_bh, _, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            print(f"    BH-FDR correct cells: {rej_bh.sum()}")

    print("\n=== v2 correct-direction cells ===")
    for _, r in df[df.ensemble == "v2"].sort_values("perm_p").iterrows():
        print(f"  {r.model[:30]:<30s} clean={r.clean_acc:.3f}(n={int(r.n_clean)}) "
              f"contam={r.contam_acc:.3f}(n={int(r.n_contam)}) gap={r.obs_gap:+.3f} p={r.perm_p:.4f}")

    print("\n=== v1 cells ===")
    for _, r in df[df.ensemble == "v1"].sort_values("perm_p").iterrows():
        print(f"  {r.model[:30]:<30s} clean={r.clean_acc:.3f}(n={int(r.n_clean)}) "
              f"contam={r.contam_acc:.3f}(n={int(r.n_contam)}) gap={r.obs_gap:+.3f} p={r.perm_p:.4f}")

    df.to_csv(f"{OUT}/mbpp_sandbox_cells.csv", index=False)
    print(f"\nSaved to {OUT}/mbpp_sandbox_cells.csv")


if __name__ == "__main__":
    main()
