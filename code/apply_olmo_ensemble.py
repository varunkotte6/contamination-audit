#!/usr/bin/env python3
"""Train a NEW v2 ensemble from Qwen-0.5B ground-truth variants and apply
to the 22-model audit matrix. Check Bonferroni survival.

R2's criterion for non-refmodel-delta-dominated Bonferroni + non-Pythia
ground-truth replication.
"""
import json
import os
import pickle
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
from statsmodels.stats.multitest import multipletests

PROJECT = "{REPO_ROOT}"
PHASE3_QWEN = f"{PROJECT}/results/phase3_olmo"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
OUT = f"{PROJECT}/results/olmo_ensemble_audit"
Path(OUT).mkdir(parents=True, exist_ok=True)

VARIANTS = ["A", "B", "C", "D", "E"]
DEV = ["A", "B", "C"]
VAL = ["D", "E"]

FEATURE_SETS = {
    "olmo_v2_full": ["ngram13_pile", "ngram13_fineweb",
                      "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
                      "loss_ft", "loss_base", "refmodel_delta"],
    "olmo_v2_no_refdelta": ["ngram13_pile", "ngram13_fineweb",
                             "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
                             "loss_ft", "loss_base"],
    "olmo_v2_pure_mem": ["ngram13_pile", "ngram13_fineweb",
                          "mink_5", "mink_10", "mink_20", "mink_30", "mink_50"],
}


def load_calib_data():
    out = {}
    for v in VARIANTS:
        p = f"{PHASE3_QWEN}/{v}/scores.pkl"
        if not os.path.exists(p):
            print(f"  [missing] {p}")
            continue
        with open(p, "rb") as f:
            out[v] = pickle.load(f)
        print(f"  loaded variant {v}: pos={int(out[v]['label'].sum())} "
              f"neg={int((out[v]['label']==0).sum())}")
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
    if not X_dev or not X_val:
        return None, None, None
    X_dev = np.vstack(X_dev); y_dev = np.concatenate(y_dev)
    X_val = np.vstack(X_val); y_val = np.concatenate(y_val)
    scaler = StandardScaler().fit(X_dev)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        scaler.transform(X_dev), y_dev)
    s_val = clf.predict_proba(scaler.transform(X_val))[:, 1]
    p, r, _ = precision_recall_curve(y_val, s_val)
    f1 = 2 * p * r / (p + r + 1e-12)
    return clf, scaler, {"f1_val": float(np.nanmax(f1)),
                          "auc_val": float(roc_auc_score(y_val, s_val)),
                          "n_dev": len(y_dev), "n_val": len(y_val)}


def apply_to_audit(clf, scaler, features):
    out = {}
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
        out[model] = {"provenance": d["provenance"], "p_contam": p_contam}
    return out


def length_quartile(items, bench):
    lengths = np.array([len(json.dumps(it)) for it in items])
    qs = np.quantile(lengths, [0.25, 0.5, 0.75])
    return np.digitize(lengths, qs)


def gsm8k_step_count(items):
    import re
    counts = np.array([len(re.findall(r"<<[^>]*>>", it.get("answer", "")))
                        for it in items])
    # Bin: 0-2, 3-4, 5-6, 7+
    bins = np.zeros(len(items), dtype=int)
    bins[counts >= 3] = 1
    bins[counts >= 5] = 2
    bins[counts >= 7] = 3
    return bins


def compute_stratified(p_by_model, name):
    UNIFORM = ["gsm8k_test", "mmlu_test", "arc_challenge", "humaneval"]
    rows = []
    for model, r in p_by_model.items():
        prov = r["provenance"]; p_contam = r["p_contam"]
        for bench in UNIFORM:
            idxs = [i for i, (b, _) in enumerate(prov) if b == bench]
            if not idxs:
                continue
            item_ids = [prov[i][1] for i in idxs]
            p = p_contam[idxs]
            order = np.argsort(item_ids)
            p = p[order]
            pi_path = f"{PHASE5_FULL}/{model}/per_item.pkl"
            if not os.path.exists(pi_path):
                continue
            with open(pi_path, "rb") as f:
                pi = pickle.load(f)
            if bench not in pi:
                continue
            correct = np.array([row["correct"] for row in pi[bench]["per_item"]],
                                dtype=np.int32)
            items = []
            with open(f"{BENCH_DIR}/{bench}.jsonl") as f:
                for line in f:
                    items.append(json.loads(line))
            n = min(len(correct), len(p), len(items))
            if n < 40: continue
            correct = correct[:n]; p = p[:n]
            # Use step-count for GSM8K, length-quartile otherwise
            if bench == "gsm8k_test":
                strata = gsm8k_step_count(items[:n])
            else:
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
            if not gaps: continue
            gaps = np.array(gaps); ses = np.array(ses)
            w = 1/ses**2
            iv_gap = (w*gaps).sum()/w.sum()
            iv_se = 1/np.sqrt(w.sum())
            z = iv_gap/iv_se
            p_corr = stats.norm.cdf(z)
            rows.append({"model": model, "benchmark": bench,
                          "iv_gap": iv_gap, "z": z, "p_correct": p_corr,
                          "n_strata": len(gaps)})
    m = len(rows)
    bonf = 0.05 / m if m > 0 else 0.05
    raw = [r for r in rows if r["p_correct"] < 0.05 and r["iv_gap"] < 0]
    bonf_c = [r for r in rows if r["p_correct"] < bonf and r["iv_gap"] < 0]
    fams = set()
    for r in bonf_c:
        m_ = r["model"].lower()
        if "llama" in m_: fams.add("Llama")
        elif "qwen__qwen2.5-32b" in m_: fams.add("Qwen-32B")
        elif "qwen__qwen2.5-14b" in m_: fams.add("Qwen-14B")
        elif "qwen__qwen3" in m_: fams.add("Qwen-3")
        elif "qwen" in m_: fams.add("Qwen-other")
        elif "olmo" in m_: fams.add("OLMo")
        elif "mistral" in m_: fams.add("Mistral")
        elif "phi-3" in m_.lower(): fams.add("Phi-3.5")
        elif "phi" in m_.lower(): fams.add("Phi-4")
        elif "gemma" in m_.lower(): fams.add("Gemma")
        else: fams.add(r["model"].split("__")[0])
    return {"m": m, "n_raw": len(raw), "n_bonf": len(bonf_c),
             "families": sorted(fams),
             "bonf_cells": [{"m": r["model"], "b": r["benchmark"],
                              "gap": r["iv_gap"], "p": r["p_correct"]}
                             for r in sorted(bonf_c, key=lambda x: x["p_correct"])]}


def main():
    print("Loading Qwen variant calibration data...")
    data = load_calib_data()
    if len(data) < 5:
        print(f"ERROR: only {len(data)} variants loaded, need 5")
        return

    results = {}
    for name, features in FEATURE_SETS.items():
        print(f"\n=== {name} ({len(features)} features) ===")
        clf, scaler, metrics = train_ensemble(data, features)
        if clf is None:
            continue
        print(f"  calibration: F1_val={metrics['f1_val']:.4f} AUC={metrics['auc_val']:.4f}")
        print(f"  applying to audit matrix...")
        p_by = apply_to_audit(clf, scaler, features)
        print(f"  scored {len(p_by)} models")
        strat = compute_stratified(p_by, name)
        print(f"  m={strat['m']} raw_sig={strat['n_raw']} bonf={strat['n_bonf']}")
        print(f"  Bonferroni families: {strat['families']}")
        for c in strat["bonf_cells"]:
            print(f"    {c['m']:<42s}  {c['b']:<18s}  gap={c['gap']:+.3f}  p={c['p']:.4f}")
        results[name] = {"calibration": metrics, "stratified": strat}

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT}/summary.json")


if __name__ == "__main__":
    main()
