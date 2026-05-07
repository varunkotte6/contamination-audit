#!/usr/bin/env python3
"""Mixed-effects robustness check for v1-vs-v2 gap distribution test.

Reviewer R4 correctly noted that the Mann-Whitney test over per-cell
gaps treats cells as i.i.d., but cells share structure through model
and benchmark. We re-test the v1-vs-v2 difference using a linear
mixed-effects model with random intercepts for model and benchmark,
and also a permutation test that shuffles calibration labels WITHIN
each (model, benchmark) pair to preserve the structure.

We use the stratified-analysis iv_weighted_gap as the outcome (the
same quantity Table 3 reports), and run on the uniform-difficulty
subset (matching the paper's headline 5-vs-0 claim).
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False

PROJECT = "{REPO_ROOT}"
STRAT_INPUT = f"{PROJECT}/results/stratified_impact/stratified_gaps.csv"
POOL_INPUT = f"{PROJECT}/results/stat_tests/stat_tests_v2.csv"
OUT = f"{PROJECT}/results/robustness"
Path(OUT).mkdir(parents=True, exist_ok=True)

UNIFORM_BENCHES = {"gsm8k_test", "mmlu_test", "hellaswag_val",
                    "arc_challenge", "humaneval", "humaneval_sandbox",
                    "mbpp", "truthfulqa_mc"}


def main():
    # First fit on the pooled (unstratified) obs_gap — broader sample
    # than stratified which requires the "both v1 and v2 cell" intersection.
    pool = pd.read_csv(POOL_INPUT)
    pool = pool[pool.ensemble.isin(["v1", "v2"])].copy()
    pool = pool[pool.benchmark.isin(UNIFORM_BENCHES)].copy()
    pool["is_v1"] = (pool.ensemble == "v1").astype(int)
    print(f"[pooled] uniform-difficulty rows: {len(pool)} across "
          f"{pool.model.nunique()} models and {pool.benchmark.nunique()} benchmarks")

    pool_summary = None
    # Approach 1: mixed-effects with random intercepts for model.
    # Known to be singular when within-model variance dominates, so we
    # report whatever succeeds and fall back to OLS with cluster-robust SE.
    if HAS_SM and len(pool) >= 10:
        import warnings
        warnings.filterwarnings("ignore")
        # Simpler model: drop benchmark fixed effects (they take too many df)
        try:
            md = smf.mixedlm("obs_gap ~ is_v1", pool, groups=pool["model"])
            mdf = md.fit(reml=False, method="lbfgs")
            beta = mdf.fe_params.get("is_v1", None)
            se = mdf.bse.get("is_v1", None)
            pval = mdf.pvalues.get("is_v1", None)
            print("\n[1] Mixed-effects (random intercept per model, "
                  "fixed: ensemble):")
            print(f"  n_obs={len(pool)}  n_groups={pool['model'].nunique()}")
            print(f"  beta(is_v1)={beta:+.4f}  se={se:.4f}  p(two-sided)={pval:.4f}")
            pool_summary = {"mixedlm_model_re": {"beta": float(beta),
                                                   "se": float(se),
                                                   "p": float(pval)}}
        except Exception as e:
            print(f"[mixedlm-model failed] {e}")

        # Second model: random intercepts for benchmark (more variance)
        try:
            md2 = smf.mixedlm("obs_gap ~ is_v1", pool,
                               groups=pool["benchmark"])
            mdf2 = md2.fit(reml=False, method="lbfgs")
            beta2 = mdf2.fe_params.get("is_v1", None)
            se2 = mdf2.bse.get("is_v1", None)
            pval2 = mdf2.pvalues.get("is_v1", None)
            print(f"\n[2] Mixed-effects (random intercept per benchmark):")
            print(f"  n_obs={len(pool)}  n_groups={pool['benchmark'].nunique()}")
            print(f"  beta(is_v1)={beta2:+.4f}  se={se2:.4f}  p(two-sided)={pval2:.4f}")
            if pool_summary is None:
                pool_summary = {}
            pool_summary["mixedlm_benchmark_re"] = {"beta": float(beta2),
                                                     "se": float(se2),
                                                     "p": float(pval2)}
        except Exception as e:
            print(f"[mixedlm-benchmark failed] {e}")

        # Crossed random effects (model + benchmark) via statsmodels
        # workaround: encode each grouping as dummies and fit with BOTH as
        # random; statsmodels MixedLM accepts a single `groups` but supports
        # multiple random effects via `vc_formula` (variance components).
        try:
            pool["one"] = 1
            vc = {"benchmark": "0 + C(benchmark)"}
            md_cross = smf.mixedlm("obs_gap ~ is_v1", pool,
                                     groups=pool["one"], re_formula="0",
                                     vc_formula=vc)
            # Refit using `groups = model` and vc = benchmark to get crossed
            md_cross = smf.mixedlm("obs_gap ~ is_v1", pool,
                                     groups=pool["model"],
                                     vc_formula={"benchmark": "0 + C(benchmark)"})
            mdf_cross = md_cross.fit(reml=False, method="lbfgs")
            beta_c = mdf_cross.fe_params.get("is_v1", None)
            se_c = mdf_cross.bse.get("is_v1", None)
            pval_c = mdf_cross.pvalues.get("is_v1", None)
            print(f"\n[crossed RE] MixedLM with crossed random effects "
                  "(model + benchmark):")
            print(f"  n_obs={len(pool)}  n_model_groups={pool['model'].nunique()}  "
                  f"n_benchmarks={pool['benchmark'].nunique()}")
            print(f"  beta(is_v1)={beta_c:+.4f}  se={se_c:.4f}  "
                  f"p(two-sided)={pval_c:.4f}")
            if pool_summary is None:
                pool_summary = {}
            pool_summary["mixedlm_crossed"] = {"beta": float(beta_c),
                                                 "se": float(se_c),
                                                 "p": float(pval_c)}
        except Exception as e:
            print(f"[crossed-RE failed] {e}")

        # OLS with cluster-robust SE (standard alternative to mixed-effects)
        try:
            import statsmodels.api as sm
            X = sm.add_constant(pool[["is_v1"]].astype(float))
            y = pool["obs_gap"].values
            ols = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": pool["model"].values})
            beta3 = ols.params["is_v1"]
            se3 = ols.bse["is_v1"]
            pval3 = ols.pvalues["is_v1"]
            print(f"\n[3] OLS with cluster-robust SE (cluster by model):")
            print(f"  beta(is_v1)={beta3:+.4f}  se(cluster)={se3:.4f}  "
                  f"p(two-sided)={pval3:.4f}")
            if pool_summary is None:
                pool_summary = {}
            pool_summary["ols_cluster_model"] = {"beta": float(beta3),
                                                  "se": float(se3),
                                                  "p": float(pval3)}
        except Exception as e:
            print(f"[OLS-cluster failed] {e}")

    df = pd.read_csv(STRAT_INPUT)
    # Uniform-difficulty subset (matches paper §5 headline regime)
    df = df[df.benchmark.isin(UNIFORM_BENCHES)].copy()
    # Only cells that exist under both ensembles so pairing is possible
    pivot = df.pivot_table(index=["model", "benchmark"],
                           columns="ensemble",
                           values="iv_weighted_gap", aggfunc="first")
    pivot_se = df.pivot_table(index=["model", "benchmark"],
                              columns="ensemble",
                              values="iv_se", aggfunc="first")
    both = pivot.dropna(subset=["v1", "v2"])
    print(f"Uniform-difficulty cells with both v1 and v2 values: {len(both)}")

    # Paired (within-cell) analyses — these fully respect the pairing
    # structure by design and sidestep the independence violation.
    diffs = both["v1"].values - both["v2"].values
    if len(diffs) >= 2:
        t_stat, t_p = stats.ttest_1samp(diffs, 0)
        wilcoxon = stats.wilcoxon(both["v1"].values, both["v2"].values,
                                   alternative="greater")
        sign_test_pos = int((diffs > 0).sum())
        sign_test_n = len(diffs)
        sign_p = stats.binomtest(sign_test_pos, sign_test_n, p=0.5,
                                  alternative="greater").pvalue
        print(f"\nPaired (within-cell) tests (v1 - v2 gap > 0 means v1 more "
              f"wrong-direction):")
        print(f"  n_pairs = {sign_test_n}")
        print(f"  mean(v1 - v2) = {diffs.mean():+.4f}")
        print(f"  paired t-test: t={t_stat:+.3f}, p(two-sided)={t_p:.4f}")
        print(f"  Wilcoxon signed-rank (one-sided v1 > v2): W={wilcoxon.statistic:.1f}, "
              f"p={wilcoxon.pvalue:.4f}")
        print(f"  Sign test (one-sided v1 > v2): {sign_test_pos}/{sign_test_n}, "
              f"p={sign_p:.4f}")

    # Mixed-effects model on the stacked long-format data
    # Model: gap ~ ensemble + (1 | model) + (1 | benchmark)
    # Implementation: statsmodels MixedLM allows one random effect; we use
    # (model) as the grouping variable and add benchmark as fixed effects,
    # since most model families appear across benchmarks.
    if HAS_SM and len(df) > 10:
        df_long = df[df.ensemble.isin(["v1", "v2"])].copy()
        df_long["is_v1"] = (df_long.ensemble == "v1").astype(int)
        try:
            md = smf.mixedlm("iv_weighted_gap ~ is_v1 + C(benchmark)",
                              df_long, groups=df_long["model"])
            mdf = md.fit(reml=False, method="lbfgs")
            print("\nMixed-effects (random intercept per model, "
                  "fixed effects: benchmark + ensemble):")
            beta = mdf.fe_params.get("is_v1", None)
            se = mdf.bse.get("is_v1", None)
            zval = mdf.tvalues.get("is_v1", None)
            pval = mdf.pvalues.get("is_v1", None)
            print(f"  n_obs = {len(df_long)}, n_groups = {df_long['model'].nunique()}")
            print(f"  ensemble=v1 coefficient: beta={beta:+.4f}  se={se:.4f}  "
                  f"z={zval:+.3f}  p(two-sided)={pval:.4f}")
        except Exception as e:
            print(f"[mixed-effects failed] {e}")

    # Stratified permutation test: shuffle v1/v2 labels within each
    # (model, benchmark) pair, preserving the pairing. This is the
    # exact test for the null "calibration choice is exchangeable
    # within cells".
    rng = np.random.default_rng(0)
    v1g = both["v1"].values
    v2g = both["v2"].values
    obs_stat = (v1g - v2g).mean()  # positive if v1 more wrong-direction
    n_perm = 10000
    extreme = 0
    for _ in range(n_perm):
        flips = rng.integers(0, 2, size=len(v1g)).astype(bool)
        a = np.where(flips, v2g, v1g)
        b = np.where(flips, v1g, v2g)
        s = (a - b).mean()
        if s >= obs_stat:
            extreme += 1
    p_paired = extreme / n_perm
    print(f"\nPaired (within-cell) sign-flip permutation test:")
    print(f"  obs mean(v1 - v2) = {obs_stat:+.4f}")
    print(f"  P(permuted >= obs) = {p_paired:.4f} (one-sided, n_perm={n_perm})")

    # Write summary
    summary = {
        "n_pairs": int(len(both)),
        "paired_t_p_two_sided": float(t_p),
        "paired_wilcoxon_p": float(wilcoxon.pvalue),
        "paired_sign_test_p": float(sign_p),
        "paired_permutation_p": float(p_paired),
        "mean_v1_minus_v2": float(diffs.mean()),
    }
    if HAS_SM:
        try:
            summary["mixedlm_beta_is_v1"] = float(beta)
            summary["mixedlm_p_is_v1"] = float(pval)
        except Exception:
            pass
    import json
    summary["pooled_mixedlm"] = pool_summary
    with open(f"{OUT}/mixed_effects_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {OUT}/mixed_effects_summary.json")


if __name__ == "__main__":
    main()
