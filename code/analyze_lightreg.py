#!/usr/bin/env python3
"""Summarize detector F1 across light-exposure regimes.

For each (variant, epochs) combo, computes per-method F1 (with auto-sign-flip)
on same-benchmark-negatives calibration data. Also fits the v2 ensemble
(saved at results/phase3_v2/ensemble.pkl) on each regime's features to
get ensemble F1 per exposure level.

Outputs figures_v2/fig5_lightreg.pdf and table5_lightreg.csv.
"""
import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT = "{REPO_ROOT}"
LR_DIR = f"{PROJECT}/results/phase3_lightreg"
V2_DIR = f"{PROJECT}/results/phase3_v2"
OUT_FIG = f"{PROJECT}/figures_v2"
Path(OUT_FIG).mkdir(parents=True, exist_ok=True)

AUDIT_FEATURES = [
    "ngram13_pile", "ngram13_fineweb",
    "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
    "loss_ft", "loss_base", "refmodel_delta",
]


def best_f1(scores, labels, flip=True):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    mask = ~np.isnan(s)
    s, y = s[mask], y[mask]
    if len(np.unique(y)) < 2 or len(s) < 10:
        return {"f1": None, "auc": None, "sign": 1.0}
    sign = 1.0
    try:
        if flip and roc_auc_score(y, s) < 0.5:
            sign = -1.0
            s = -s
    except Exception:
        pass
    thrs = np.percentile(s, np.linspace(0, 100, 201))
    best = -1.0
    for t in thrs:
        pred = (s >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if f1 > best:
            best = f1
    try:
        auc = float(roc_auc_score(y, s))
    except Exception:
        auc = None
    return {"f1": float(best), "auc": auc, "sign": sign}


def ensemble_f1(scores_dict, methods=AUDIT_FEATURES):
    """Fit a logistic regression on the 10-feature audit set, report F1 on same data."""
    labels = scores_dict["label"]
    n = len(labels)
    cols, keep = [], np.ones(n, dtype=bool)
    for m in methods:
        if m not in scores_dict:
            return None
        col = np.asarray(scores_dict[m], dtype=np.float64)
        keep &= ~np.isnan(col)
        cols.append(col)
    X = np.stack(cols, axis=1)[keep]
    y = labels[keep]
    if len(np.unique(y)) < 2 or X.shape[0] < 20:
        return None
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X), y)
    pred = clf.predict(scaler.transform(X))
    _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(y, clf.predict_proba(scaler.transform(X))[:, 1]))
    except Exception:
        auc = None
    return {"f1": float(f1), "auc": auc}


def main():
    rows = []
    # Light-regime (1/5/10 epoch)
    for variant in ["A", "B", "C", "D"]:
        for ep in [1, 5, 10]:
            p = f"{LR_DIR}/{variant}_{ep}ep/scores.pkl"
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                s = pickle.load(f)
            row = {"variant": variant, "epochs": ep}
            for m in AUDIT_FEATURES:
                r = best_f1(s.get(m, np.array([])), s["label"])
                row[f"{m}_f1"] = r["f1"]
                row[f"{m}_auc"] = r["auc"]
            ens = ensemble_f1(s)
            row["ensemble_f1"] = ens["f1"] if ens else None
            row["ensemble_auc"] = ens["auc"] if ens else None
            vb_path = f"{PROJECT}/checkpoints/gt_{variant}_{ep}ep/metrics.json"
            if os.path.exists(vb_path):
                row["verbatim_rate"] = json.load(open(vb_path))["verbatim_rate"]
            rows.append(row)

        # 50-epoch from phase3_v2
        p = f"{V2_DIR}/{variant}/scores.pkl"
        if os.path.exists(p):
            with open(p, "rb") as f:
                s = pickle.load(f)
            row = {"variant": variant, "epochs": 50}
            for m in AUDIT_FEATURES:
                r = best_f1(s.get(m, np.array([])), s["label"])
                row[f"{m}_f1"] = r["f1"]; row[f"{m}_auc"] = r["auc"]
            ens = ensemble_f1(s)
            row["ensemble_f1"] = ens["f1"] if ens else None
            row["ensemble_auc"] = ens["auc"] if ens else None
            vb_path = f"{PROJECT}/checkpoints/gt_{variant}/metrics.json"
            if os.path.exists(vb_path):
                row["verbatim_rate"] = json.load(open(vb_path))["verbatim_rate"]
            rows.append(row)

    if not rows:
        raise SystemExit("No light-regime scores found.")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_FIG}/table5_lightreg.csv", index=False)
    print(df[["variant", "epochs", "verbatim_rate", "ensemble_f1",
               "ensemble_auc", "mink_20_f1", "refmodel_delta_f1"]].to_string(index=False))

    # Plot: F1 vs exposure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for variant, g in df.groupby("variant"):
        g = g.sort_values("epochs")
        axes[0].plot(g["epochs"], g["ensemble_f1"], "o-", label=f"Variant {variant}")
        axes[1].plot(g["epochs"], g["verbatim_rate"], "s-", label=f"Variant {variant}")
    for ax in axes:
        ax.set_xlabel("Fine-tune epochs (exposure)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xscale("log")
    axes[0].set_ylabel("Ensemble F1 (v2 calibration scheme)")
    axes[0].set_title("Detection F1 vs exposure")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("Verbatim reproduction rate")
    axes[1].set_title("Memorization vs exposure")
    axes[1].set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(f"{OUT_FIG}/fig5_lightreg.pdf")
    plt.savefig(f"{OUT_FIG}/fig5_lightreg.png", dpi=150)
    print(f"\n[SAVED] {OUT_FIG}/fig5_lightreg.pdf, table5_lightreg.csv")


if __name__ == "__main__":
    main()
