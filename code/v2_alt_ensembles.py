#!/usr/bin/env python3
"""Alternative v2 ensembles: train without refmodel_delta (R2's critique).

R2 explicitly said moving from 4→5 requires "Bonferroni survival across ≥3
families with a non-refmodel_delta-dominated ensemble". We train 2 variants:

  v2-no-refdelta: drop refmodel_delta only; keep loss_ft, loss_base as
    separate features.
  v2-pure-mem: only mink_* and ngram_* features; no loss features at all.

For each, we re-fit the LR ensemble on the 5 Pythia-1B variants, then
re-apply to the 22-model audit matrix and recompute stratified Bonferroni
survival. Report the per-ensemble cell counts and family counts.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

PROJECT = "{REPO_ROOT}"
PHASE3_V2 = f"{PROJECT}/results/phase3_v2"
PHASE4_V1 = f"{PROJECT}/results/phase4"
OUT = f"{PROJECT}/results/v2_alt_ensembles"
Path(OUT).mkdir(parents=True, exist_ok=True)

VARIANTS = ["A", "B", "C", "D", "E"]
DEV = ["A", "B", "C"]
VAL = ["D", "E"]

FEATURE_SETS = {
    "v2_full": ["ngram13_pile", "ngram13_fineweb",
                 "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
                 "loss_ft", "loss_base", "refmodel_delta"],
    "v2_no_refdelta": ["ngram13_pile", "ngram13_fineweb",
                        "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
                        "loss_ft", "loss_base"],
    "v2_pure_mem": ["ngram13_pile", "ngram13_fineweb",
                     "mink_5", "mink_10", "mink_20", "mink_30", "mink_50"],
}


def load_calib_data():
    """Load calibration features + labels from Pythia-1B variants."""
    out = {}
    for v in VARIANTS:
        p = f"{PHASE3_V2}/{v}/scores.pkl"
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            out[v] = pickle.load(f)
    return out


def train_ensemble(data, features):
    X_dev, y_dev, X_val, y_val = [], [], [], []
    for v, d in data.items():
        X = np.column_stack([d[f] for f in features])
        X = np.nan_to_num(X)
        y = d["label"]
        if v in DEV:
            X_dev.append(X); y_dev.append(y)
        else:
            X_val.append(X); y_val.append(y)
    X_dev = np.vstack(X_dev); y_dev = np.concatenate(y_dev)
    X_val = np.vstack(X_val); y_val = np.concatenate(y_val)
    scaler = StandardScaler().fit(X_dev)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        scaler.transform(X_dev), y_dev)
    # F1 on val
    s_val = clf.predict_proba(scaler.transform(X_val))[:, 1]
    p, r, _ = precision_recall_curve(y_val, s_val)
    f1 = 2 * p * r / (p + r + 1e-12)
    f1_best = float(np.nanmax(f1))
    auc = float(roc_auc_score(y_val, s_val))
    return clf, scaler, {"f1_val": f1_best, "auc_val": auc,
                          "n_dev": len(y_dev), "n_val": len(y_val)}


def apply_to_audit(clf, scaler, features):
    """Apply ensemble to 22 audit models' v2 features, compute p_contam."""
    results = {}
    for model in sorted(os.listdir(PHASE4_V1)):
        if model.startswith("_"):
            continue
        sp = f"{PHASE4_V1}/{model}/scores.pkl"
        if not os.path.exists(sp):
            continue
        with open(sp, "rb") as f:
            d = pickle.load(f)
        feats = d["features"]
        if not all(f in feats for f in features):
            continue
        X = np.column_stack([feats[f] for f in features])
        X = np.nan_to_num(X)
        p_contam = clf.predict_proba(scaler.transform(X))[:, 1]
        results[model] = {"provenance": d["provenance"], "p_contam": p_contam}
    return results


def compute_stratified_bonferroni(p_contam_by_model, features_name):
    """Compute Bonferroni-surviving count using length-quartile stratification
    for uniform-difficulty benchmarks."""
    import json
    from scipy import stats
    BENCH_DIR = "/mnt/localssd/data/benchmarks"
    UNIFORM = ["gsm8k_test", "mmlu_test", "arc_challenge", "humaneval"]
    PHASE5 = f"{PROJECT}/results/phase5_full"

    def load_accuracy(model, bench):
        pi_path = f"{PHASE5}/{model}/per_item.pkl"
        if not os.path.exists(pi_path):
            return None
        with open(pi_path, "rb") as f:
            pi = pickle.load(f)
        if bench not in pi:
            return None
        return np.array([r["correct"] for r in pi[bench]["per_item"]], dtype=np.int32)

    def load_bench_items(bench):
        path = f"{BENCH_DIR}/{bench}.jsonl"
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f]

    def length_quartile(items, bench):
        lengths = np.array([len(json.dumps(it)) for it in items])
        qs = np.quantile(lengths, [0.25, 0.5, 0.75])
        return np.digitize(lengths, qs)

    rows = []
    for model, r in p_contam_by_model.items():
        prov = r["provenance"]; p_contam = r["p_contam"]
        for bench in UNIFORM:
            idxs = [i for i, (b, _) in enumerate(prov) if b == bench]
            if not idxs:
                continue
            item_ids = [prov[i][1] for i in idxs]
            p = p_contam[idxs]
            order = np.argsort(item_ids)
            p = p[order]
            correct = load_accuracy(model, bench)
            if correct is None:
                continue
            items = load_bench_items(bench)
            if not items:
                continue
            n = min(len(correct), len(p), len(items))
            if n < 40: continue
            correct = correct[:n]; p = p[:n]
            strata = length_quartile(items[:n], bench)
            # Stratified gap
            gaps, ses = [], []
            for b in range(4):
                m = strata == b
                cc = correct[m & (p < 0.1)]
                ct = correct[m & (p > 0.5)]
                if len(cc) < 5 or len(ct) < 5:
                    continue
                gap = cc.mean() - ct.mean()
                se = np.sqrt(cc.var(ddof=1)/len(cc) + ct.var(ddof=1)/len(ct))
                if se < 1e-6 or not np.isfinite(se): continue
                gaps.append(gap); ses.append(se)
            if len(gaps) < 1: continue
            gaps = np.array(gaps); ses = np.array(ses)
            w = 1/ses**2
            iv_gap = (w*gaps).sum()/w.sum()
            iv_se = 1/np.sqrt(w.sum())
            z = iv_gap/iv_se
            p_correct = stats.norm.cdf(z)
            p_two = 2*min(p_correct, 1-p_correct)
            rows.append({"model": model, "benchmark": bench,
                          "iv_gap": iv_gap, "z": z, "p_correct": p_correct,
                          "p_two": p_two})
    # Bonferroni at m = len(rows)
    m = len(rows)
    bonf = 0.05 / m if m > 0 else 0.05
    bonf_cells = [r for r in rows if r["p_correct"] < bonf and r["iv_gap"] < 0]
    raw_cells = [r for r in rows if r["p_correct"] < 0.05 and r["iv_gap"] < 0]
    families = set()
    for r in bonf_cells:
        m_ = r["model"].lower()
        if "llama" in m_: families.add("llama")
        elif "qwen2.5-32b" in m_.lower(): families.add("qwen-32b")
        elif "qwen2.5-14b" in m_.lower(): families.add("qwen-14b")
        elif "qwen" in m_.lower(): families.add("qwen-other")
        elif "olmo" in m_.lower(): families.add("olmo")
        elif "mistral" in m_: families.add("mistral")
        else: families.add(r["model"].split("/")[0])
    return {"name": features_name, "m": m, "n_raw": len(raw_cells),
             "n_bonf": len(bonf_cells), "families_bonf": sorted(families),
             "bonf_cells": [{"m": r["model"], "b": r["benchmark"],
                              "gap": r["iv_gap"], "p": r["p_correct"]}
                             for r in bonf_cells]}


def main():
    data = load_calib_data()
    print(f"Loaded {len(data)} variants: {list(data.keys())}")

    summary = {}
    for name, features in FEATURE_SETS.items():
        print(f"\n=== Training {name} ensemble ({len(features)} features) ===")
        clf, scaler, metrics = train_ensemble(data, features)
        print(f"  F1_val={metrics['f1_val']:.4f}  AUC_val={metrics['auc_val']:.4f}")

        print(f"  Applying to audit matrix...")
        p_by_model = apply_to_audit(clf, scaler, features)
        print(f"  Scored {len(p_by_model)} models")

        print(f"  Computing stratified Bonferroni...")
        strat_result = compute_stratified_bonferroni(p_by_model, name)
        print(f"  n_tests={strat_result['m']}  raw_sig={strat_result['n_raw']}  "
              f"bonf_sig={strat_result['n_bonf']}  "
              f"families={strat_result['families_bonf']}")
        for c in strat_result["bonf_cells"]:
            print(f"    {c['m']:<35s}  {c['b']:<18s}  gap={c['gap']:+.3f}  p={c['p']:.4f}")

        summary[name] = {
            "calibration": metrics,
            "stratified": strat_result,
        }

    import json
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {OUT}/summary.json")


if __name__ == "__main__":
    main()
