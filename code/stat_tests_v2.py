#!/usr/bin/env python3
"""Statistical tests on clean-vs-contaminated accuracy gaps, expanded to
include MATH-500 and sandboxed HumanEval / LiveCodeBench.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = "{REPO_ROOT}"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
MATH_EVAL = f"{PROJECT}/results/math_eval"
HEVAL_SANDBOX = f"{PROJECT}/results/humaneval_sandbox"
LCB_SANDBOX = f"{PROJECT}/results/livecodebench_sandbox"
OUT = f"{PROJECT}/results/stat_tests"
Path(OUT).mkdir(parents=True, exist_ok=True)

BENCHMARKS_PHASE5 = ["gsm8k_test", "mmlu_test", "hellaswag_val",
                      "arc_challenge", "humaneval", "mbpp", "truthfulqa_mc"]


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
    return {
        "n_clean": nc, "n_contam": nt,
        "clean_acc": float(cc.mean()), "contam_acc": float(ct.mean()),
        "obs_gap": float(gap), "ci_low": ci_lo, "ci_high": ci_hi,
        "perm_p": float(perm_p),
    }


def main():
    rows = []

    # Phase 5 full: GSM8K, MMLU, HellaSwag, ARC, HumanEval (prefix-match)
    for m in sorted(os.listdir(PHASE5_FULL)):
        if m.startswith("_"):
            continue
        pi_path = f"{PHASE5_FULL}/{m}/per_item.pkl"
        if not os.path.exists(pi_path):
            continue
        with open(pi_path, "rb") as f:
            per_item = pickle.load(f)
        for bench in BENCHMARKS_PHASE5:
            if bench not in per_item:
                continue
            correct = np.array([r["correct"] for r in per_item[bench]["per_item"]], dtype=np.int32)
            for version, ph4 in [("v1", PHASE4_V1), ("v2", PHASE4_V2)]:
                p = load_p_contam(ph4, m, bench)
                if p is None:
                    continue
                N = min(len(correct), len(p))
                stat = cell_stat(correct[:N], p[:N])
                if stat is None:
                    continue
                rows.append({
                    "model": m.replace("__", "/"),
                    "benchmark": bench, "ensemble": version,
                    "evaluator": "phase5_full",
                    **stat,
                })

    # MATH-500 (numeric match)
    for d_name in sorted(os.listdir(MATH_EVAL)):
        if not d_name.endswith("_math500"):
            continue
        model = d_name[:-len("_math500")]
        rf = f"{MATH_EVAL}/{d_name}/results.pkl"
        if not os.path.exists(rf):
            continue
        with open(rf, "rb") as f:
            results = pickle.load(f)
        correct = np.array([r["correct"] for r in results], dtype=np.int32)
        for version, ph4 in [("v1", PHASE4_V1), ("v2", PHASE4_V2)]:
            p = load_p_contam(ph4, model, "math500")
            if p is None:
                continue
            N = min(len(correct), len(p))
            stat = cell_stat(correct[:N], p[:N])
            if stat is None:
                continue
            rows.append({
                "model": model.replace("__", "/"),
                "benchmark": "math500", "ensemble": version,
                "evaluator": "numeric_boxed",
                **stat,
            })

    # HumanEval sandboxed
    for m in sorted(os.listdir(HEVAL_SANDBOX)):
        if m.startswith("_"):
            continue
        rf = f"{HEVAL_SANDBOX}/{m}/results.pkl"
        if not os.path.exists(rf):
            continue
        with open(rf, "rb") as f:
            results = pickle.load(f)
        correct = np.array([int(r["passed"]) for r in results], dtype=np.int32)
        for version, ph4 in [("v1", PHASE4_V1), ("v2", PHASE4_V2)]:
            p = load_p_contam(ph4, m, "humaneval")
            if p is None:
                continue
            N = min(len(correct), len(p))
            stat = cell_stat(correct[:N], p[:N])
            if stat is None:
                continue
            rows.append({
                "model": m.replace("__", "/"),
                "benchmark": "humaneval_sandbox", "ensemble": version,
                "evaluator": "sandbox_pass1",
                **stat,
            })

    # LiveCodeBench sandboxed
    for m in sorted(os.listdir(LCB_SANDBOX)):
        if m.startswith("_"):
            continue
        rf = f"{LCB_SANDBOX}/{m}/results.pkl"
        if not os.path.exists(rf):
            continue
        with open(rf, "rb") as f:
            results = pickle.load(f)
        correct = np.array([int(r["passed"]) for r in results], dtype=np.int32)
        for version, ph4 in [("v1", PHASE4_V1), ("v2", PHASE4_V2)]:
            p = load_p_contam(ph4, m, "livecodebench")
            if p is None:
                continue
            N = min(len(correct), len(p))
            stat = cell_stat(correct[:N], p[:N])
            if stat is None:
                continue
            rows.append({
                "model": m.replace("__", "/"),
                "benchmark": "livecodebench_sandbox", "ensemble": version,
                "evaluator": "sandbox_pass1",
                **stat,
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/stat_tests_v2.csv", index=False)
    print(f"Total cells: {len(df)}")

    for ens in ["v1", "v2"]:
        sub = df[df.ensemble == ens]
        if len(sub) == 0:
            continue
        n_sig_correct = ((sub.obs_gap < 0) & (sub.perm_p < 0.05)).sum()
        n_sig_wrong = ((sub.obs_gap > 0) & (sub.perm_p > 0.95)).sum()
        print(f"  {ens}: n={len(sub)}  mean_gap={sub.obs_gap.mean():+.4f}  "
              f"sig_correct={n_sig_correct}  sig_wrong={n_sig_wrong}")
        # Aggregate t-test
        g = sub.obs_gap.values
        t, p_two = stats.ttest_1samp(g, 0)
        print(f"    one-sample t vs 0: t={t:+.3f}, two-sided p={p_two:.4f}")
    # v1 vs v2
    v1g = df[df.ensemble == "v1"].obs_gap.values
    v2g = df[df.ensemble == "v2"].obs_gap.values
    if len(v1g) and len(v2g):
        t, p = stats.ttest_ind(v1g, v2g, equal_var=False)
        u, p_u = stats.mannwhitneyu(v1g, v2g, alternative="greater")
        print(f"\n  v1 vs v2: t-test p={p:.4f}, Mann-Whitney (v1 > v2) p={p_u:.4f}")

    # Significantly inverted cells
    print("\n=== v1 significantly inverted cells ===")
    v1w = df[(df.ensemble == "v1") & (df.obs_gap > 0) & (df.perm_p > 0.95)].sort_values("obs_gap", ascending=False)
    for _, r in v1w.head(10).iterrows():
        print(f"  {r.model[:35]:<35s} {r.benchmark[:25]:<25s} clean={r.clean_acc:.3f}(n={int(r.n_clean)}) contam={r.contam_acc:.3f}(n={int(r.n_contam)}) gap={r.obs_gap:+.3f} p={r.perm_p:.3f}")
    print("\n=== v2 significantly correct cells ===")
    v2c = df[(df.ensemble == "v2") & (df.obs_gap < 0) & (df.perm_p < 0.05)].sort_values("obs_gap")
    for _, r in v2c.iterrows():
        print(f"  {r.model[:35]:<35s} {r.benchmark[:25]:<25s} clean={r.clean_acc:.3f}(n={int(r.n_clean)}) contam={r.contam_acc:.3f}(n={int(r.n_contam)}) gap={r.obs_gap:+.3f} p={r.perm_p:.3f}")


if __name__ == "__main__":
    main()
