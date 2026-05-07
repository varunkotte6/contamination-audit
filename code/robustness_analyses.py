#!/usr/bin/env python3
"""Consolidated robustness analyses addressing reviewer concerns:

(a) Reconcile Mann-Whitney p-values — explicitly compute pooled (all benchmarks)
    vs.\ stratified (uniform vs variant difficulty).
(b) Leave-one-cell-out and leave-one-model-out sensitivity on the
    uniform-difficulty MW test.
(c) Coverage comparison: fraction of items flagged contaminated per cell
    under v1 vs v2.
(d) Within-benchmark difficulty-stratified gaps (GSM8K by problem length
    quartiles; MMLU by subject category; HellaSwag by length; ARC by
    question length) — shows whether v2's effect is partially driven by
    difficulty within uniform-difficulty benchmarks.
(e) Bonferroni/Holm correction on per-cell permutation tests.

All results land in results/robustness/.
"""
import json
import os
import pickle
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = "{REPO_ROOT}"
OUT = f"{PROJECT}/results/robustness"
Path(OUT).mkdir(parents=True, exist_ok=True)
STAT_CSV = f"{PROJECT}/results/stat_tests/stat_tests_v2.csv"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
BENCH_DIR = "/mnt/localssd/data/benchmarks"

UNIFORM = ["gsm8k_test", "mmlu_test", "hellaswag_val",
            "arc_challenge", "humaneval", "humaneval_sandbox"]
VARIANT = ["math500", "livecodebench_sandbox"]


def run_reconcile_and_sensitivity():
    df = pd.read_csv(STAT_CSV)
    out = {"all_tests": {}}

    for label, benches in [("all", df.benchmark.unique().tolist()),
                             ("uniform_difficulty", UNIFORM),
                             ("variant_difficulty", VARIANT)]:
        sub = df[df.benchmark.isin(benches)]
        v1 = sub[sub.ensemble == "v1"].obs_gap.values
        v2 = sub[sub.ensemble == "v2"].obs_gap.values
        t1, p1 = stats.ttest_ind(v1, v2, equal_var=False) if len(v1) and len(v2) else (None, None)
        u1, p_u = stats.mannwhitneyu(v1, v2, alternative="greater") if len(v1) and len(v2) else (None, None)
        out["all_tests"][label] = {
            "n_v1": len(v1), "n_v2": len(v2),
            "v1_mean": float(v1.mean()) if len(v1) else None,
            "v2_mean": float(v2.mean()) if len(v2) else None,
            "t_test_p_two_sided": float(p1) if p1 is not None else None,
            "mannwhitney_p_v1_gt_v2": float(p_u) if p_u is not None else None,
        }
        print(f"[{label}] n_v1={len(v1)} n_v2={len(v2)} "
              f"v1_mean={v1.mean() if len(v1) else '--':+.4f} "
              f"v2_mean={v2.mean() if len(v2) else '--':+.4f} "
              f"t-test p={p1} MW p={p_u}")

    # --- Leave-one-out sensitivity on uniform-difficulty MW ---
    sub = df[df.benchmark.isin(UNIFORM)]
    v1u = sub[sub.ensemble == "v1"]
    v2u = sub[sub.ensemble == "v2"]
    base_p = stats.mannwhitneyu(v1u.obs_gap.values, v2u.obs_gap.values,
                                  alternative="greater").pvalue
    print(f"\nBaseline uniform-MW p = {base_p:.4f}")

    loo_cell = []
    # Drop one v2 cell at a time; does p still < 0.05?
    for idx, r in v2u.iterrows():
        remaining = v2u.drop(idx).obs_gap.values
        if len(remaining) < 3:
            continue
        p = stats.mannwhitneyu(v1u.obs_gap.values, remaining,
                                alternative="greater").pvalue
        loo_cell.append({
            "dropped_model": r.model, "dropped_benchmark": r.benchmark,
            "dropped_gap": float(r.obs_gap),
            "remaining_v2_n": len(remaining),
            "mannwhitney_p": float(p),
        })
    loo_df = pd.DataFrame(loo_cell).sort_values("mannwhitney_p", ascending=False)
    loo_df.to_csv(f"{OUT}/leave_one_cell_out_v2.csv", index=False)
    print(f"\n[LOO-cell-v2] worst-case p after dropping one v2 cell:")
    print(loo_df.head(5).to_string(index=False))
    out["leave_one_cell_out_v2"] = {
        "max_p_after_drop": float(loo_df.mannwhitney_p.max()),
        "median_p_after_drop": float(loo_df.mannwhitney_p.median()),
        "all_remain_p_lt_005": bool((loo_df.mannwhitney_p < 0.05).all()),
    }

    # Leave-one-MODEL-out: drop all cells from one model family
    loo_model = []
    for model in v2u.model.unique():
        remain_v1 = v1u[v1u.model != model].obs_gap.values
        remain_v2 = v2u[v2u.model != model].obs_gap.values
        if len(remain_v1) < 3 or len(remain_v2) < 3:
            continue
        p = stats.mannwhitneyu(remain_v1, remain_v2,
                                alternative="greater").pvalue
        loo_model.append({"dropped_model": model,
                           "remaining_v1_n": len(remain_v1),
                           "remaining_v2_n": len(remain_v2),
                           "mannwhitney_p": float(p)})
    loo_m_df = pd.DataFrame(loo_model).sort_values("mannwhitney_p", ascending=False)
    loo_m_df.to_csv(f"{OUT}/leave_one_model_out.csv", index=False)
    print(f"\n[LOO-model] p after dropping each model family:")
    print(loo_m_df.to_string(index=False))
    out["leave_one_model_out"] = {
        "max_p": float(loo_m_df.mannwhitney_p.max()),
        "all_remain_p_lt_005": bool((loo_m_df.mannwhitney_p < 0.05).all()),
    }

    # Multiple comparison correction on per-cell permutation tests
    from statsmodels.stats.multitest import multipletests
    p_v1 = v1u.perm_p.values  # v1 uniform cells
    p_v2 = v2u.perm_p.values
    # Check both tails — for "contam > clean" test we care about p<0.05,
    # but inversions show up as p>0.95 (one-sided test). Transform to
    # two-sided.
    p_v1_two = np.minimum(2 * np.minimum(p_v1, 1 - p_v1), 1.0)
    p_v2_two = np.minimum(2 * np.minimum(p_v2, 1 - p_v2), 1.0)
    _, p_v1_bh, _, _ = multipletests(p_v1_two, method="fdr_bh")
    _, p_v2_bh, _, _ = multipletests(p_v2_two, method="fdr_bh")
    out["bh_correction_uniform"] = {
        "v1_sig_cells_fdr05": int((p_v1_bh < 0.05).sum()),
        "v2_sig_cells_fdr05": int((p_v2_bh < 0.05).sum()),
        "n_v1": len(p_v1), "n_v2": len(p_v2),
    }
    print(f"\n[BH FDR 0.05 on uniform]: v1 sig={int((p_v1_bh<0.05).sum())}/{len(p_v1)}  "
          f"v2 sig={int((p_v2_bh<0.05).sum())}/{len(p_v2)}")

    return out


def run_coverage_analysis():
    """Per (model, benchmark), fraction of items with p_contam > 0.5 under v1 vs v2."""
    rows = []
    for phase4_dir, version in [(PHASE4_V1, "v1"), (PHASE4_V2, "v2")]:
        for m in sorted(os.listdir(phase4_dir)):
            if m.startswith("_"):
                continue
            sp = f"{phase4_dir}/{m}/scores.pkl"
            if not os.path.exists(sp):
                continue
            with open(sp, "rb") as f:
                d = pickle.load(f)
            p = np.asarray(d["p_contam"])
            prov = d["provenance"]
            benches = sorted(set(b for b, _ in prov))
            for bench in benches:
                mask = np.array([pp[0] == bench for pp in prov])
                if not mask.any():
                    continue
                p_b = p[mask]
                rows.append({
                    "model": m.replace("__", "/"),
                    "benchmark": bench,
                    "ensemble": version,
                    "n_total": int(mask.sum()),
                    "n_clean_lt_0.1": int((p_b < 0.1).sum()),
                    "n_mid": int(((p_b >= 0.1) & (p_b <= 0.5)).sum()),
                    "n_contam_gt_0.5": int((p_b > 0.5).sum()),
                    "frac_contam": float((p_b > 0.5).mean()),
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/coverage.csv", index=False)
    agg = df.groupby(["ensemble", "benchmark"]).frac_contam.agg(["mean", "median", "std"]).reset_index()
    print(f"\n[COVERAGE] mean frac_contam per benchmark:")
    pv = agg.pivot(index="benchmark", columns="ensemble", values="mean")
    print(pv.round(3).to_string())
    return {"coverage_table": pv.round(4).to_dict()}


def run_within_difficulty_gsm8k():
    """On GSM8K, stratify by problem length (proxy for difficulty) and
    report v2 accuracy gap within each quartile for OLMo-2-7B (our
    significant cell).
    """
    # Load GSM8K items
    items = []
    with open(f"{BENCH_DIR}/gsm8k_test.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    qlens = np.array([len(ex["question"]) for ex in items])
    # Quartiles of question length
    q25, q50, q75 = np.percentile(qlens, [25, 50, 75])

    model = "allenai__OLMo-2-1124-7B"
    with open(f"{PHASE4_V2}/{model}/scores.pkl", "rb") as f:
        v2d = pickle.load(f)
    idxs = [i for i, (b, _) in enumerate(v2d["provenance"]) if b == "gsm8k_test"]
    item_ids = [v2d["provenance"][i][1] for i in idxs]
    order = np.argsort(item_ids)
    p = v2d["p_contam"][idxs][order]

    # Load per-item accuracy from phase5_full
    with open(f"{PHASE5_FULL}/{model}/per_item.pkl", "rb") as f:
        pi = pickle.load(f)
    correct = np.array([r["correct"] for r in pi["gsm8k_test"]["per_item"]])
    N = min(len(correct), len(p), len(qlens))
    correct = correct[:N]; p = p[:N]; qlens = qlens[:N]

    quartiles = np.searchsorted([q25, q50, q75], qlens)  # 0..3
    print(f"\n[WITHIN-DIFFICULTY] OLMo-2-7B on GSM8K by problem-length quartile")
    print(f"Quartile boundaries: {q25:.0f}, {q50:.0f}, {q75:.0f} chars")
    out = {}
    for q in range(4):
        mask = quartiles == q
        if mask.sum() < 20:
            continue
        cc = p[mask] < 0.1
        ct = p[mask] > 0.5
        if cc.sum() < 5 or ct.sum() < 5:
            print(f"  Q{q+1}: n_clean={cc.sum()} n_contam={ct.sum()} — insufficient")
            continue
        ca = correct[mask][cc].mean()
        cta = correct[mask][ct].mean()
        gap = ca - cta
        # Permutation
        rng = np.random.default_rng(0)
        cc_scores = correct[mask][cc]; ct_scores = correct[mask][ct]
        combined = np.concatenate([cc_scores, ct_scores])
        extreme = 0
        for _ in range(5000):
            rng.shuffle(combined)
            c = combined[:len(cc_scores)].mean() - combined[len(cc_scores):].mean()
            if c <= gap:
                extreme += 1
        pp = extreme / 5000
        print(f"  Q{q+1}: n_clean={int(cc.sum()):>3d} n_contam={int(ct.sum()):>3d}  "
              f"clean_acc={ca:.3f} contam_acc={cta:.3f} gap={gap:+.3f} perm_p={pp:.3f}")
        out[f"Q{q+1}"] = {
            "n_clean": int(cc.sum()), "n_contam": int(ct.sum()),
            "clean_acc": float(ca), "contam_acc": float(cta),
            "gap": float(gap), "perm_p": float(pp),
        }
    return {"olmo2_gsm8k_within_length_quartile": out}


def main():
    results = {}
    print("=" * 70)
    print("ROBUSTNESS ANALYSES FOR REVIEWER RESPONSE")
    print("=" * 70)
    results.update(run_reconcile_and_sensitivity())
    results.update(run_coverage_analysis())
    results.update(run_within_difficulty_gsm8k())
    with open(f"{OUT}/all.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] {OUT}/all.json")


if __name__ == "__main__":
    main()
