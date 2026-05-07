#!/usr/bin/env python3
"""Feature ablation on the v2 ensemble.

For each feature, train the ensemble without it and report the change
in validation F1 and in correct-direction deployment rate. Identifies
which features are load-bearing for the v2 fix.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT = "{REPO_ROOT}"
V2_DIR = f"{PROJECT}/results/phase3_v2"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
OUT_DIR = f"{PROJECT}/figures_v2"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

FULL_FEATURES = [
    "ngram13_pile", "ngram13_fineweb",
    "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
    "loss_ft", "loss_base", "refmodel_delta",
]
DEV = ["A", "B", "C"]
VAL = ["D", "E"]
BENCHMARKS = ["gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge", "humaneval"]


def load_scores_for(variant):
    p = f"{V2_DIR}/{variant}/scores.pkl"
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def fit_ensemble(features):
    """Train v2 ensemble on A,B,C and evaluate on D,E."""
    def stack(variants):
        X, y = [], []
        for v in variants:
            s = load_scores_for(v)
            if s is None:
                continue
            n = len(s["label"])
            keep = np.ones(n, dtype=bool)
            cols = []
            for m in features:
                if m not in s:
                    return None, None, None
                col = np.asarray(s[m], dtype=np.float64)
                keep &= ~np.isnan(col)
                cols.append(col)
            X_v = np.stack(cols, axis=1)[keep]
            y_v = s["label"][keep]
            X.append(X_v); y.append(y_v)
        return np.concatenate(X), np.concatenate(y), features

    X_dev, y_dev, _ = stack(DEV)
    X_val, y_val, _ = stack(VAL)
    if X_dev is None or len(X_dev) < 20:
        return None

    scaler = StandardScaler().fit(X_dev)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_dev), y_dev)
    val_p = clf.predict_proba(scaler.transform(X_val))[:, 1]
    val_pred = (val_p >= 0.5).astype(int)
    pp, rr, f1_val, _ = precision_recall_fscore_support(y_val, val_pred, average="binary", zero_division=0)
    try:
        auc_val = float(roc_auc_score(y_val, val_p))
    except Exception:
        auc_val = None
    return {
        "features": features,
        "val_f1": float(f1_val), "val_auc": auc_val,
        "scaler": scaler, "clf": clf, "methods": features,
    }


def correct_direction_rate(ensemble):
    """Apply this ensemble to phase4_v1 features and slice phase5_full
    predictions; report fraction of (model, bench) cells with contam > clean
    at n >= 20 in both subsets."""
    methods = ensemble["methods"]
    scaler = ensemble["scaler"]
    clf = ensemble["clf"]

    gap_data = []
    for m in sorted(os.listdir(PHASE4_V1)):
        if m.startswith("_"):
            continue
        sp = f"{PHASE4_V1}/{m}/scores.pkl"
        pp = f"{PHASE5_FULL}/{m}/per_item.pkl"
        if not (os.path.exists(sp) and os.path.exists(pp)):
            continue
        with open(sp, "rb") as f:
            fd = pickle.load(f)
        feats = dict(fd["features"])
        if "loss_audit" in feats and "loss_ft" not in feats:
            feats["loss_ft"] = feats.pop("loss_audit")
        if "loss_ref" in feats and "loss_base" not in feats:
            feats["loss_base"] = feats.pop("loss_ref")

        n_rows = len(fd["p_contam"])
        cols = []
        for me in methods:
            if me in feats:
                cols.append(np.asarray(feats[me], dtype=np.float64))
            else:
                cols.append(np.full(n_rows, np.nan))
        X = np.stack(cols, axis=1)
        for j in range(X.shape[1]):
            col = X[:, j]
            nan = np.isnan(col)
            if nan.any():
                col[nan] = scaler.mean_[j]
        p_contam = clf.predict_proba(scaler.transform(X))[:, 1]

        with open(pp, "rb") as f:
            per_item = pickle.load(f)
        prov = fd["provenance"]
        for bench in BENCHMARKS:
            if bench not in per_item:
                continue
            correct = np.array([int(r["correct"]) for r in per_item[bench]["per_item"]])
            idxs = [i for i, (bb, _) in enumerate(prov) if bb == bench]
            if not idxs:
                continue
            item_ids = [prov[i][1] for i in idxs]
            p = p_contam[idxs]
            order = np.argsort(item_ids)
            p = p[order]
            N = min(len(correct), len(p))
            correct = correct[:N]; p = p[:N]
            clean_mask = p < 0.1
            contam_mask = p > 0.5
            if clean_mask.sum() < 20 or contam_mask.sum() < 20:
                continue
            clean_acc = float(correct[clean_mask].mean())
            contam_acc = float(correct[contam_mask].mean())
            gap_data.append({
                "model": m, "benchmark": bench,
                "clean_n": int(clean_mask.sum()), "contam_n": int(contam_mask.sum()),
                "clean_acc": clean_acc, "contam_acc": contam_acc,
                "clean_minus_contam": clean_acc - contam_acc,
            })
    if not gap_data:
        return {"n_cells": 0, "mean_gap": None, "correct_dir_pct": None}
    df = pd.DataFrame(gap_data)
    n_correct = (df.clean_minus_contam < 0).sum()
    return {
        "n_cells": len(df),
        "mean_gap": float(df.clean_minus_contam.mean()),
        "correct_dir_pct": float(n_correct / len(df)),
        "cells": df,
    }


def main():
    # Baseline full ensemble
    print("Full ensemble (all 10 features):")
    baseline = fit_ensemble(FULL_FEATURES)
    if baseline is None:
        raise SystemExit("Could not fit baseline.")
    base_corr = correct_direction_rate(baseline)
    print(f"  val_F1={baseline['val_f1']:.4f}  val_AUC={baseline['val_auc']:.4f}")
    print(f"  correct-direction cells (n>=20): {base_corr['correct_dir_pct']:.1%}  "
          f"mean_gap={base_corr['mean_gap']:+.4f}  n={base_corr['n_cells']}")

    rows = []
    rows.append({
        "dropped": "(none)",
        "val_f1": baseline["val_f1"], "val_auc": baseline["val_auc"],
        "correct_dir_pct": base_corr["correct_dir_pct"],
        "mean_gap_pp": base_corr["mean_gap"] * 100 if base_corr["mean_gap"] is not None else None,
        "n_cells": base_corr["n_cells"],
    })

    # Leave-one-out
    print("\nLeave-one-out:")
    for drop_feat in FULL_FEATURES:
        remaining = [f for f in FULL_FEATURES if f != drop_feat]
        ens = fit_ensemble(remaining)
        if ens is None:
            continue
        corr = correct_direction_rate(ens)
        rows.append({
            "dropped": drop_feat,
            "val_f1": ens["val_f1"], "val_auc": ens["val_auc"],
            "correct_dir_pct": corr["correct_dir_pct"],
            "mean_gap_pp": corr["mean_gap"] * 100 if corr["mean_gap"] is not None else None,
            "n_cells": corr["n_cells"],
        })
        print(f"  drop {drop_feat:<18s} val_F1={ens['val_f1']:.4f}  "
              f"correct_dir={(corr['correct_dir_pct'] or 0):.1%}  "
              f"gap={(corr['mean_gap'] or 0)*100:+.2f}pp")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/table_ablation.csv", index=False)
    with open(f"{OUT_DIR}/table_ablation.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))
    print(f"\n[SAVED] {OUT_DIR}/table_ablation.csv")


if __name__ == "__main__":
    main()
