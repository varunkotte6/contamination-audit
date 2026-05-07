#!/usr/bin/env python3
"""Power analysis + wild-cluster bootstrap + MDE funnel plot.

For each v2 cell that did NOT reach correct-direction significance,
compute the minimum detectable effect (MDE) at 80% power given the
observed n_clean, n_contam, and per-cell variance. Build a funnel
plot and report heterogeneity (I^2, tau^2).

Also runs a wild-cluster bootstrap with cluster-by-model to supplement
the cluster-robust SE result.
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT = "{REPO_ROOT}"
OUT = f"{PROJECT}/results/power_analysis"
Path(OUT).mkdir(parents=True, exist_ok=True)


def mde_power(n1, n2, p_baseline, power=0.80, alpha=0.05):
    """Minimum detectable effect (absolute gap in proportions) for a
    two-sample proportion test at given power."""
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha)
    z_b = norm.ppf(power)
    # Approximate SE for pooled Bernoulli difference
    p = p_baseline
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (z_a + z_b) * se


def main():
    df = pd.read_csv(f"{PROJECT}/results/stratified_impact/stratified_gaps.csv")
    v2 = df[df.ensemble == "v2"].copy()
    UNIFORM = {"gsm8k_test", "mmlu_test", "hellaswag_val",
                "arc_challenge", "humaneval", "humaneval_sandbox",
                "mbpp", "truthfulqa_mc"}
    v2 = v2[v2.benchmark.isin(UNIFORM)].copy()
    print(f"v2 uniform cells: {len(v2)}")

    # For each cell, compute MDE at 80% power given baseline accuracy p
    # We need cell-level n_clean, n_contam, and baseline accuracy. These aren't
    # in stratified_gaps.csv (it has n_strata, iv_weighted_gap, iv_se). Instead,
    # read the per-cell stat_tests_v2.csv which has n_clean, n_contam, clean_acc.
    cells = pd.read_csv(f"{PROJECT}/results/stat_tests/stat_tests_v2.csv")
    cells = cells[cells.ensemble == "v2"]
    cells_uniform = cells[cells.benchmark.isin(UNIFORM)].copy()
    cells = cells_uniform
    rows = []
    for _, r in cells.iterrows():
        nc = int(r.n_clean); nt = int(r.n_contam)
        p = float(r.clean_acc)
        mde = mde_power(nc, nt, p) if nc > 0 and nt > 0 else float("nan")
        rows.append({"model": r.model, "benchmark": r.benchmark,
                     "n_clean": nc, "n_contam": nt,
                     "observed_gap": float(r.obs_gap),
                     "clean_acc": p,
                     "MDE_80power": float(mde),
                     "abs_obs_gap": float(abs(r.obs_gap)),
                     "reached_MDE": float(abs(r.obs_gap) >= mde)})
    mdf = pd.DataFrame(rows)
    mdf.to_csv(f"{OUT}/mde_per_cell.csv", index=False)
    print("\n=== MDE per v2 cell (uniform-difficulty) ===")
    print(f"{'model':<28s} {'benchmark':<18s} {'n_cl':>4s} {'n_ct':>4s} "
          f"{'obs_gap':>8s} {'MDE_80%':>8s} {'reached':>8s}")
    for _, r in mdf.iterrows():
        print(f"{r.model[:28]:<28s} {r.benchmark[:18]:<18s} "
              f"{int(r.n_clean):>4d} {int(r.n_contam):>4d} "
              f"{r.observed_gap:>+8.3f} {r.MDE_80power:>8.3f} "
              f"{'Y' if r.reached_MDE else 'N':>8s}")

    # Funnel plot data: cell gap vs SE (1/sqrt(n_contam))
    mdf["se_approx"] = np.sqrt(mdf.clean_acc * (1 - mdf.clean_acc)
                                * (1 / mdf.n_clean + 1 / mdf.n_contam))
    mdf.to_csv(f"{OUT}/funnel_data.csv", index=False)

    # Heterogeneity: I^2 and tau^2 over the v2 cells (drop NaN rows)
    mdf_hq = mdf.dropna(subset=["observed_gap", "se_approx"]).copy()
    mdf_hq = mdf_hq[mdf_hq.se_approx > 1e-6]
    n = len(mdf_hq)
    if n > 2:
        gaps = mdf_hq.observed_gap.values
        ses = mdf_hq.se_approx.values
        w = 1 / ses**2
        # Fixed-effect weighted mean
        mean_FE = (w * gaps).sum() / w.sum()
        # Cochran's Q
        Q = ((w * (gaps - mean_FE)**2)).sum()
        # Degrees of freedom
        dfree = n - 1
        I2 = max(0.0, 1 - dfree / Q) if Q > 0 else 0.0
        # tau^2 (DerSimonian-Laird)
        C = w.sum() - (w**2).sum() / w.sum()
        tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
        print(f"\nHeterogeneity (v2 uniform cells):")
        print(f"  Q = {Q:.3f} (df={dfree})")
        print(f"  I^2 = {I2:.3f}")
        print(f"  tau^2 = {tau2:.6f}")
        # DerSimonian-Laird pooled estimate under RE
        w_re = 1 / (ses**2 + tau2)
        mean_RE = (w_re * gaps).sum() / w_re.sum()
        se_RE = 1 / np.sqrt(w_re.sum())
        z_RE = mean_RE / se_RE
        p_RE_two = 2 * stats.norm.cdf(-abs(z_RE))
        p_RE_correct = stats.norm.cdf(z_RE)  # P(gap <= 0)
        print(f"  DL-pooled mean (RE): {mean_RE:+.4f} ± {se_RE:.4f}  "
              f"z = {z_RE:+.3f}  p(two-sided) = {p_RE_two:.4f}  "
              f"p(correct-direction) = {p_RE_correct:.4f}")

    # Wild-cluster bootstrap for v1-vs-v2 contrast (cluster by model)
    print("\n=== Wild-cluster bootstrap (cluster by model) ===")
    pool = cells.copy()
    pool_v1 = pd.read_csv(f"{PROJECT}/results/stat_tests/stat_tests_v2.csv")
    pool_all = pool_v1[pool_v1.ensemble.isin(["v1", "v2"])
                        & pool_v1.benchmark.isin(UNIFORM)].copy()
    pool_all["is_v1"] = (pool_all.ensemble == "v1").astype(int)
    pool_all["resid"] = (pool_all.obs_gap
                          - pool_all.groupby("model").obs_gap.transform("mean"))
    # Observed t-stat under OLS: is_v1 coefficient / cluster-robust SE
    import statsmodels.api as sm
    X = sm.add_constant(pool_all[["is_v1"]].astype(float))
    y = pool_all.obs_gap.values
    ols = sm.OLS(y, X).fit(cov_type="cluster",
                            cov_kwds={"groups": pool_all["model"].values})
    t_obs = ols.tvalues["is_v1"]
    beta_obs = ols.params["is_v1"]
    print(f"  obs beta_is_v1 = {beta_obs:+.4f}, t = {t_obs:+.3f}")

    # Wild-cluster bootstrap under the null: flip Rademacher weights per cluster
    n_boot = 2000
    clusters = pool_all["model"].unique()
    rng = np.random.default_rng(0)
    t_boot = np.empty(n_boot)
    for b in range(n_boot):
        flips = rng.choice([-1, 1], size=len(clusters))
        flip_map = dict(zip(clusters, flips))
        # y_star = y + flip[cluster] * resid  (Rademacher wild bootstrap)
        y_star = y + np.array([flip_map[m] for m in pool_all["model"]]) * pool_all["resid"].values
        X_b = sm.add_constant(pool_all[["is_v1"]].astype(float))
        try:
            ols_b = sm.OLS(y_star, X_b).fit(
                cov_type="cluster",
                cov_kwds={"groups": pool_all["model"].values})
            t_boot[b] = ols_b.tvalues["is_v1"]
        except Exception:
            t_boot[b] = np.nan
    valid = ~np.isnan(t_boot)
    t_boot = t_boot[valid]
    # Two-sided p
    p_two = np.mean(np.abs(t_boot) >= np.abs(t_obs))
    p_one = np.mean(t_boot >= t_obs)
    print(f"  n_boot valid: {len(t_boot)}")
    print(f"  Wild-cluster bootstrap p (two-sided): {p_two:.4f}")
    print(f"  Wild-cluster bootstrap p (one-sided, v1>v2): {p_one:.4f}")

    # Hartung-Knapp adjustment for small-K random-effects meta-analysis
    # (R4 specifically requested this instead of DL)
    # H-K uses a modified variance of pooled effect + t-distribution with K-1 df
    if n > 2:
        # Effective weights under RE
        w_re_hk = 1 / (ses**2 + tau2)
        mean_hk = (w_re_hk * gaps).sum() / w_re_hk.sum()
        # H-K variance: empirical weighted residual variance
        q_star = ((w_re_hk * (gaps - mean_hk)**2).sum()) / (n - 1)
        var_hk = q_star / w_re_hk.sum()
        se_hk = np.sqrt(var_hk)
        t_hk = mean_hk / se_hk
        p_hk_two = 2 * stats.t.sf(abs(t_hk), df=n - 1)
        p_hk_correct = stats.t.cdf(t_hk, df=n - 1)  # one-sided correct
        print(f"\nHartung-Knapp meta-analysis (small-K correction):")
        print(f"  n = {n}, df = {n - 1}")
        print(f"  mean_HK = {mean_hk:+.4f}  se_HK = {se_hk:.4f}  t = {t_hk:+.3f}  "
              f"p(two-sided) = {p_hk_two:.4f}  p(correct) = {p_hk_correct:.4f}")

    summary = {
        "heterogeneity": {"I2": float(I2), "tau2": float(tau2), "Q": float(Q)},
        "DL_pooled": {"mean": float(mean_RE), "se": float(se_RE),
                       "p_two_sided": float(p_RE_two)},
        "hartung_knapp": {"mean": float(mean_hk) if n > 2 else None,
                          "se": float(se_hk) if n > 2 else None,
                          "t": float(t_hk) if n > 2 else None,
                          "p_two_sided": float(p_hk_two) if n > 2 else None,
                          "p_correct_direction": float(p_hk_correct) if n > 2 else None,
                          "df": n - 1 if n > 2 else None},
        "wild_cluster_bootstrap": {
            "obs_t": float(t_obs), "n_boot": int(len(t_boot)),
            "p_two_sided": float(p_two), "p_one_sided": float(p_one),
        },
    }
    with open(f"{OUT}/power_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT}/power_summary.json")


if __name__ == "__main__":
    main()
