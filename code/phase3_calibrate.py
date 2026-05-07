#!/usr/bin/env python3
"""Phase 3 — Calibration and ensemble training.

Loads detector scores from results/phase3/{A..E}/scores.pkl and:
  1. For each individual detector, sweeps threshold and reports
     precision/recall/F1 and ROC-AUC. Optimal threshold chosen on A+B+C
     (development set); D+E used for held-out validation.
  2. Trains a logistic-regression ensemble on standardized detector scores
     using variants A+B+C, evaluated on D+E. Reports ensemble F1 and
     inter-detector Cohen's kappa matrix.

Writes:
  results/phase3/calibration.json — per-method metrics + chosen thresholds
  results/phase3/ensemble.pkl     — trained logistic regression (scaler + model)
  results/phase3/method_roc.csv    — ROC points per method for plotting
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT = "{REPO_ROOT}"
RESULTS = f"{PROJECT}/results/phase3"

DEV = ["A", "B", "C"]
VAL = ["D", "E"]
ALL = DEV + VAL


def load_scores():
    """Return dict[variant -> dict[method -> array]]."""
    out = {}
    for v in ALL:
        path = f"{RESULTS}/{v}/scores.pkl"
        if not os.path.exists(path):
            print(f"[WARN] missing {path}, skipping variant {v}")
            continue
        with open(path, "rb") as f:
            out[v] = pickle.load(f)
    return out


def best_threshold(scores, labels):
    """Find threshold maximizing F1 over `labels in {0,1}` positive-class.
    Returns (best_thr, best_f1, precision, recall)."""
    # Candidate thresholds = unique values in scores
    thrs = np.unique(scores)
    if len(thrs) > 500:
        # Downsample for speed
        thrs = np.percentile(scores, np.linspace(0, 100, 501))
    best = (None, -1.0, 0, 0)
    for thr in thrs:
        pred = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
        if f1 > best[1]:
            best = (float(thr), float(f1), float(p), float(r))
    return best


def per_method_calibration(all_scores):
    """For each method, pool dev-set scores, pick best threshold, evaluate on val.

    Auto-detects sign via AUC: if AUC<0.5, flip the score so that higher always
    means more contaminated (required for the threshold sweep to work)."""
    methods = [k for k in next(iter(all_scores.values())).keys() if k != "label"]
    out = {}
    for m in methods:
        dev_s, dev_l, val_s, val_l = [], [], [], []
        for v, s in all_scores.items():
            if m not in s:
                continue
            # Drop NaNs
            mask = ~np.isnan(s[m])
            if v in DEV:
                dev_s.append(s[m][mask]); dev_l.append(s["label"][mask])
            else:
                val_s.append(s[m][mask]); val_l.append(s["label"][mask])
        if not dev_s:
            continue
        dev_s = np.concatenate(dev_s); dev_l = np.concatenate(dev_l)
        val_s = np.concatenate(val_s) if val_s else np.array([])
        val_l = np.concatenate(val_l) if val_l else np.array([])

        # Auto-flip if AUC<0.5 so HIGH = more contaminated
        sign = 1.0
        try:
            if len(np.unique(dev_l)) == 2 and roc_auc_score(dev_l, dev_s) < 0.5:
                sign = -1.0
                dev_s = -dev_s
                if len(val_s):
                    val_s = -val_s
        except Exception:
            pass

        thr, f1_dev, p_dev, r_dev = best_threshold(dev_s, dev_l)

        # Evaluate on val
        if len(val_l) and len(np.unique(val_l)) == 2:
            pred_val = (val_s >= thr).astype(int)
            p_val, r_val, f1_val, _ = precision_recall_fscore_support(
                val_l, pred_val, average="binary", zero_division=0)
            auc_val = float(roc_auc_score(val_l, val_s))
        else:
            p_val = r_val = f1_val = auc_val = None

        try:
            auc_dev = float(roc_auc_score(dev_l, dev_s))
        except Exception:
            auc_dev = None

        out[m] = {
            "threshold": thr,
            "sign": sign,
            "dev": {"f1": f1_dev, "precision": p_dev, "recall": r_dev,
                     "auc": auc_dev, "n": int(len(dev_l))},
            "val": {"f1": f1_val, "precision": p_val, "recall": r_val,
                     "auc": auc_val, "n": int(len(val_l))},
        }
        print(f"[{m}] sign={'+' if sign>0 else '-'} dev_f1={f1_dev:.3f} "
              f"dev_auc={auc_dev or 'n/a'} | "
              f"val_f1={f1_val if f1_val is None else f'{f1_val:.3f}'} "
              f"val_auc={auc_val if auc_val is None else f'{auc_val:.3f}'}  thr={thr:.4f}")
    return out


def inter_method_kappa(all_scores, calibration):
    """Pairwise Cohen's kappa across methods using each method's thresholded predictions
    on the whole dev set."""
    methods = list(calibration.keys())
    preds = {}
    labels_by_method = {}
    for m in methods:
        thr = calibration[m]["threshold"]
        sign = calibration[m].get("sign", 1.0)
        agg = []; lab = []
        for v in DEV:
            if v not in all_scores or m not in all_scores[v]:
                continue
            mask = ~np.isnan(all_scores[v][m])
            agg.append(((sign * all_scores[v][m][mask]) >= thr).astype(int))
            lab.append(all_scores[v]["label"][mask])
        if agg:
            preds[m] = np.concatenate(agg)
            labels_by_method[m] = np.concatenate(lab)

    kappa = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            if m1 in preds and m2 in preds and len(preds[m1]) == len(preds[m2]):
                kappa[f"{m1}__vs__{m2}"] = float(cohen_kappa_score(preds[m1], preds[m2]))
    return kappa


AUDIT_FEATURES = [
    "ngram13_pile", "ngram13_fineweb",
    "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
    "loss_ft",     # in audit, this is the audited model's loss (renamed loss_audit downstream)
    "loss_base",   # reference Pythia-1B
    "refmodel_delta",
]


def train_ensemble(all_scores, calibration, methods=None):
    """Train logistic regression on standardized detector scores.

    If `methods` is None, uses all calibration methods (includes leak features
    like ngram13_self, embsim_self that don't exist in audit). If `methods` is
    a list, restricts the feature set — used for the audit-deployable ensemble."""
    if methods is None:
        methods = list(calibration.keys())
    else:
        methods = [m for m in methods if m in calibration]

    def stack(variants):
        X, y = [], []
        for v in variants:
            if v not in all_scores:
                continue
            s = all_scores[v]
            # Need all methods present and aligned
            if not all(m in s for m in methods):
                continue
            # Drop rows with NaN in any method
            n = len(s["label"])
            keep = np.ones(n, dtype=bool)
            cols = []
            for m in methods:
                keep &= ~np.isnan(s[m])
                cols.append(s[m])
            X_v = np.stack(cols, axis=1)[keep]
            y_v = s["label"][keep]
            X.append(X_v); y.append(y_v)
        return (np.concatenate(X), np.concatenate(y)) if X else (np.empty((0, len(methods))), np.empty(0))

    X_dev, y_dev = stack(DEV)
    X_val, y_val = stack(VAL)
    print(f"ensemble: dev={X_dev.shape}, val={X_val.shape}")

    scaler = StandardScaler().fit(X_dev)
    X_dev_s = scaler.transform(X_dev)
    X_val_s = scaler.transform(X_val) if len(X_val) else None

    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_dev_s, y_dev)
    p_dev = clf.predict_proba(X_dev_s)[:, 1]
    auc_dev = float(roc_auc_score(y_dev, p_dev))
    pred_dev = (p_dev >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_dev, pred_dev, average="binary", zero_division=0)
    dev_metrics = {"f1": float(f1), "precision": float(p), "recall": float(r), "auc": auc_dev, "n": int(len(y_dev))}

    val_metrics = None
    if X_val_s is not None and len(y_val) and len(np.unique(y_val)) == 2:
        p_val = clf.predict_proba(X_val_s)[:, 1]
        auc_val = float(roc_auc_score(y_val, p_val))
        pred_val = (p_val >= 0.5).astype(int)
        pp, rr, ff, _ = precision_recall_fscore_support(y_val, pred_val, average="binary", zero_division=0)
        val_metrics = {"f1": float(ff), "precision": float(pp), "recall": float(rr),
                        "auc": auc_val, "n": int(len(y_val))}

    print(f"ensemble dev: {dev_metrics}")
    print(f"ensemble val: {val_metrics}")

    return {"scaler": scaler, "clf": clf, "methods": methods,
            "dev_metrics": dev_metrics, "val_metrics": val_metrics}


def main():
    all_scores = load_scores()
    if not all_scores:
        raise SystemExit("No scores found; run phase3_detect.py for each variant first.")

    print("== Per-method calibration ==")
    calib = per_method_calibration(all_scores)

    print("\n== Inter-method kappa ==")
    kappa = inter_method_kappa(all_scores, calib)
    for k, v in sorted(kappa.items(), key=lambda x: x[1]):
        print(f"  {k}: {v:.3f}")

    print("\n== Ensemble (full features, includes leak features) ==")
    ens_full = train_ensemble(all_scores, calib)

    print("\n== Ensemble (audit-deployable features only) ==")
    ens_audit = train_ensemble(all_scores, calib, methods=AUDIT_FEATURES)

    out = {"per_method": calib, "kappa": kappa,
           "ensemble_full_dev": ens_full["dev_metrics"], "ensemble_full_val": ens_full["val_metrics"],
           "ensemble_audit_dev": ens_audit["dev_metrics"], "ensemble_audit_val": ens_audit["val_metrics"],
           "audit_features": AUDIT_FEATURES}
    with open(f"{RESULTS}/calibration.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(f"{RESULTS}/ensemble.pkl", "wb") as f:
        pickle.dump({"scaler": ens_audit["scaler"], "clf": ens_audit["clf"],
                      "methods": ens_audit["methods"],
                      "signs": {m: calib[m]["sign"] for m in ens_audit["methods"] if m in calib}},
                     f)
    with open(f"{RESULTS}/ensemble_full.pkl", "wb") as f:
        pickle.dump({"scaler": ens_full["scaler"], "clf": ens_full["clf"],
                      "methods": ens_full["methods"],
                      "signs": {m: calib[m]["sign"] for m in ens_full["methods"] if m in calib}},
                     f)
    print(f"\n[SAVED] {RESULTS}/calibration.json")
    print(f"[SAVED] {RESULTS}/ensemble.pkl (audit-deployable, {len(ens_audit['methods'])} features)")
    print(f"[SAVED] {RESULTS}/ensemble_full.pkl (all features, {len(ens_full['methods'])} features)")


if __name__ == "__main__":
    main()
