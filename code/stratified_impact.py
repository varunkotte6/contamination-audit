#!/usr/bin/env python3
"""Difficulty-stratified impact analysis.

For each (model, benchmark) cell, partition items into difficulty strata
(using benchmark-specific proxies) and compute the clean-vs-contaminated
accuracy gap WITHIN each stratum. Aggregate via inverse-variance-weighted
mean gap. This controls for the difficulty confound identified in the
MATH/LCB pooled analysis.

Difficulty proxies:
  - MATH / MATH-500: `level` field (1-5)
  - LiveCodeBench: `difficulty` field (easy/medium/hard), or `contest_date` bins
  - GSM8K: item-length quartile (characters of question)
  - MMLU: `subject` groups (STEM / humanities / social / other)
  - HellaSwag: `ctx` length quartile
  - ARC: `question` length quartile
  - HumanEval: prompt-length quartile
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
MATH_EVAL = f"{PROJECT}/results/math_eval"
HEVAL_SANDBOX = f"{PROJECT}/results/humaneval_sandbox"
LCB_SANDBOX = f"{PROJECT}/results/livecodebench_sandbox"
OUT = f"{PROJECT}/results/stratified_impact"
Path(OUT).mkdir(parents=True, exist_ok=True)


def load_items(bench_name):
    path = f"{BENCH_DIR}/{bench_name}.jsonl"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def difficulty_strata(bench_name, items):
    """Return int[N] stratum id per item, number of strata."""
    N = len(items)
    if N == 0:
        return np.array([], dtype=int), 0

    if bench_name in ("math", "math500"):
        levels = np.array([int(ex.get("level", 0)) for ex in items])
        # Normalize to 0..K-1
        uniq = sorted(set(levels.tolist()))
        idx = {v: i for i, v in enumerate(uniq)}
        return np.array([idx[v] for v in levels]), len(uniq)

    if bench_name == "livecodebench":
        diffs = np.array([ex.get("difficulty", "unknown") for ex in items])
        uniq = sorted(set(diffs.tolist()))
        idx = {v: i for i, v in enumerate(uniq)}
        return np.array([idx[v] for v in diffs]), len(uniq)

    if bench_name == "mmlu_test":
        SUBJECTS = {
            # STEM
            "college_mathematics":0, "high_school_mathematics":0, "abstract_algebra":0,
            "college_physics":0, "high_school_physics":0, "conceptual_physics":0,
            "college_chemistry":0, "high_school_chemistry":0,
            "college_biology":0, "high_school_biology":0,
            "college_computer_science":0, "high_school_computer_science":0,
            "computer_security":0, "machine_learning":0, "electrical_engineering":0,
            "astronomy":0, "anatomy":0, "college_medicine":0, "medical_genetics":0,
            "clinical_knowledge":0, "nutrition":0, "human_aging":0,
            # default others to 1
        }
        strata = np.array([SUBJECTS.get(ex.get("subject", ""), 1) for ex in items])
        return strata, 2

    # Length-based fallback for GSM8K, HellaSwag, ARC, HumanEval, HumanEval_sandbox
    if bench_name in ("gsm8k_test",):
        lengths = np.array([len(ex.get("question", "")) for ex in items])
    elif bench_name in ("hellaswag_val",):
        lengths = np.array([len(ex.get("ctx", "")) for ex in items])
    elif bench_name in ("arc_challenge",):
        lengths = np.array([len(ex.get("question", "")) for ex in items])
    elif bench_name in ("humaneval",):
        lengths = np.array([len(ex.get("prompt", "")) for ex in items])
    else:
        return np.zeros(N, dtype=int), 1

    q25, q50, q75 = np.percentile(lengths, [25, 50, 75])
    return np.searchsorted([q25, q50, q75], lengths), 4


def permutation_p(cc, ct, n_perm=5000, seed=0):
    """One-sided: p(contam > clean) under label exchange."""
    rng = np.random.default_rng(seed)
    cc = np.asarray(cc); ct = np.asarray(ct)
    if len(cc) < 3 or len(ct) < 3:
        return None, None
    obs = cc.mean() - ct.mean()
    combined = np.concatenate([cc, ct])
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        c = combined[:len(cc)].mean() - combined[len(cc):].mean()
        if c <= obs:
            extreme += 1
    return float(obs), extreme / n_perm


def bootstrap_ci(cc, ct, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    cc = np.asarray(cc); ct = np.asarray(ct)
    if len(cc) < 3 or len(ct) < 3:
        return None, None
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(cc, size=len(cc), replace=True).mean()
        b = rng.choice(ct, size=len(ct), replace=True).mean()
        gaps[i] = a - b
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def load_p_contam(phase4_dir, model, bench_name, n_items):
    sp = f"{phase4_dir}/{model}/scores.pkl"
    if not os.path.exists(sp):
        return None
    with open(sp, "rb") as f:
        d = pickle.load(f)
    idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench_name]
    if not idxs:
        return None
    item_ids = [d["provenance"][i][1] for i in idxs]
    p = d["p_contam"][idxs]
    order = np.argsort(item_ids)
    # Pad to n_items with NaN if needed
    p_sorted = p[order]
    if len(p_sorted) < n_items:
        p_out = np.full(n_items, np.nan, dtype=np.float32)
        p_out[:len(p_sorted)] = p_sorted
        return p_out
    return p_sorted[:n_items]


def stratified_cell(correct, p_contam, strata, min_stratum_n=10):
    """For each stratum with >=min_stratum_n items and both clean+contam present,
    compute the within-stratum clean-contam gap. Aggregate via inverse-variance
    weighting on pooled bootstrap variance."""
    stratum_ids = np.unique(strata)
    per = []
    for sid in stratum_ids:
        m = strata == sid
        if m.sum() < min_stratum_n:
            continue
        c = correct[m]
        p = p_contam[m]
        valid = ~np.isnan(p)
        c = c[valid]; p = p[valid]
        clean_m = p < 0.1
        contam_m = p > 0.5
        if clean_m.sum() < 5 or contam_m.sum() < 5:
            continue
        cc = c[clean_m]; ct = c[contam_m]
        obs_gap, perm_p = permutation_p(cc, ct)
        if obs_gap is None:
            continue
        # Variance estimate (asymptotic difference-of-means)
        var = (cc.var(ddof=1) / len(cc) + ct.var(ddof=1) / len(ct))
        if var <= 0:
            continue
        per.append({
            "stratum": int(sid),
            "n_clean": int(clean_m.sum()), "n_contam": int(contam_m.sum()),
            "clean_acc": float(cc.mean()), "contam_acc": float(ct.mean()),
            "gap": obs_gap, "perm_p": perm_p, "variance": float(var),
        })
    if not per:
        return None
    # Inverse-variance weighted mean gap
    weights = np.array([1 / r["variance"] for r in per])
    gaps = np.array([r["gap"] for r in per])
    iv_gap = float((weights * gaps).sum() / weights.sum())
    iv_se = float(np.sqrt(1 / weights.sum()))
    # Overall z-test against 0 (two-sided)
    z = iv_gap / iv_se if iv_se > 0 else 0
    p_two = float(2 * (1 - stats.norm.cdf(abs(z))))
    # One-sided p(contam > clean) = p(gap < 0)
    p_one_correct = float(stats.norm.cdf(z))
    return {
        "n_strata": len(per),
        "iv_weighted_gap": iv_gap,
        "iv_se": iv_se,
        "z": z,
        "p_two_sided": p_two,
        "p_one_sided_correct_direction": p_one_correct,
        "strata_detail": per,
    }


def collect_benchmark_data():
    """Returns dict[(model, benchmark)] -> (correct array, items list, evaluator)."""
    data = {}
    # Phase 5 full
    for m in sorted(os.listdir(PHASE5_FULL)):
        if m.startswith("_"):
            continue
        pi = f"{PHASE5_FULL}/{m}/per_item.pkl"
        if not os.path.exists(pi):
            continue
        with open(pi, "rb") as f:
            per_item = pickle.load(f)
        for bench in ["gsm8k_test", "mmlu_test", "hellaswag_val",
                      "arc_challenge", "humaneval"]:
            if bench not in per_item:
                continue
            correct = np.array([r["correct"] for r in per_item[bench]["per_item"]])
            data[(m, bench)] = (correct, load_items(bench), "phase5_full")
    # HumanEval sandbox
    for m in sorted(os.listdir(HEVAL_SANDBOX)):
        if m.startswith("_"):
            continue
        rf = f"{HEVAL_SANDBOX}/{m}/results.pkl"
        if not os.path.exists(rf):
            continue
        with open(rf, "rb") as f:
            res = pickle.load(f)
        correct = np.array([int(r["passed"]) for r in res])
        data[(m, "humaneval_sandbox")] = (correct, load_items("humaneval"), "sandbox_pass1")
    # MATH-500
    for d_name in sorted(os.listdir(MATH_EVAL)):
        if not d_name.endswith("_math500"):
            continue
        m = d_name[:-len("_math500")]
        rf = f"{MATH_EVAL}/{d_name}/results.pkl"
        if not os.path.exists(rf):
            continue
        with open(rf, "rb") as f:
            res = pickle.load(f)
        correct = np.array([r["correct"] for r in res])
        data[(m, "math500")] = (correct, load_items("math500"), "numeric_boxed")
    # LiveCodeBench sandbox
    for m in sorted(os.listdir(LCB_SANDBOX)):
        if m.startswith("_"):
            continue
        rf = f"{LCB_SANDBOX}/{m}/results.pkl"
        if not os.path.exists(rf):
            continue
        with open(rf, "rb") as f:
            res = pickle.load(f)
        correct = np.array([int(r["passed"]) for r in res])
        data[(m, "livecodebench_sandbox")] = (correct, load_items("livecodebench"), "sandbox_pass1")
    return data


def main():
    data = collect_benchmark_data()
    print(f"Collected {len(data)} (model, benchmark) cells")
    rows = []
    for (model, bench), (correct, items, evaluator) in data.items():
        # The benchmark name we use for p_contam lookup is the RAW benchmark
        # (strip any _sandbox suffix since phase4 stored under raw bench names)
        raw_bench = bench.replace("_sandbox", "")
        strata, n_strata = difficulty_strata(raw_bench, items)
        if n_strata <= 1:
            continue
        N = min(len(correct), len(strata), len(items))
        correct = correct[:N]; strata = strata[:N]
        for version, ph4 in [("v1", PHASE4_V1), ("v2", PHASE4_V2)]:
            p = load_p_contam(ph4, model, raw_bench, N)
            if p is None:
                continue
            result = stratified_cell(correct, p, strata)
            if result is None:
                continue
            rows.append({
                "model": model.replace("__", "/"),
                "benchmark": bench,
                "ensemble": version,
                "evaluator": evaluator,
                "n_strata": result["n_strata"],
                "iv_weighted_gap": result["iv_weighted_gap"],
                "iv_se": result["iv_se"],
                "z_stat": result["z"],
                "p_two_sided": result["p_two_sided"],
                "p_correct_direction": result["p_one_sided_correct_direction"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/stratified_gaps.csv", index=False)
    print(f"\n[SAVED] {OUT}/stratified_gaps.csv ({len(df)} cells)")

    # Aggregate: is v2+stratified mean gap significantly negative?
    UNIFORM = ["gsm8k_test", "mmlu_test", "hellaswag_val",
                "arc_challenge", "humaneval", "humaneval_sandbox"]
    VARIANT = ["math500", "livecodebench_sandbox"]

    for label, blist in [("UNIFORM", UNIFORM), ("VARIANT", VARIANT), ("ALL", None)]:
        sub = df if blist is None else df[df.benchmark.isin(blist)]
        for ens in ["v1", "v2"]:
            s = sub[sub.ensemble == ens]
            if len(s) == 0:
                continue
            g = s.iv_weighted_gap.values
            n_correct = (g < 0).sum()
            # One-sample t vs 0
            t, p = stats.ttest_1samp(g, 0)
            print(f"  {label:<8s} {ens}: n_cells={len(s):>3d}  "
                  f"mean_iv_gap={g.mean():+.4f}  "
                  f"correct_dir={n_correct}/{len(s)}  "
                  f"t_test p={p:.4f}")
        # v1 vs v2 MW
        v1g = sub[sub.ensemble == "v1"].iv_weighted_gap.values
        v2g = sub[sub.ensemble == "v2"].iv_weighted_gap.values
        if len(v1g) and len(v2g):
            u, p_u = stats.mannwhitneyu(v1g, v2g, alternative="greater")
            print(f"  {label:<8s} v1 vs v2 (MW greater): p={p_u:.4f}")

    # Cells in correct direction with p_two_sided < 0.05
    sig = df[(df.iv_weighted_gap < 0) & (df.p_two_sided < 0.05)]
    print(f"\n=== v2 cells with stratified correct-direction p<0.05 (two-sided) ===")
    sig_v2 = sig[sig.ensemble == "v2"].sort_values("p_two_sided")
    for _, r in sig_v2.iterrows():
        print(f"  {r.model[:35]:<35s} {r.benchmark:<20s} strata={int(r.n_strata)} "
              f"iv_gap={r.iv_weighted_gap:+.4f} p={r.p_two_sided:.4f}")


if __name__ == "__main__":
    main()
