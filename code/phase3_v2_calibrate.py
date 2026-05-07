#!/usr/bin/env python3
"""Phase 3 v2 — calibration against same-benchmark negatives.

Mirrors phase3_calibrate.py but reads from results/phase3_v2/ and writes to the
same directory. Also produces a side-by-side v1-vs-v2 comparison file so the
impact of the negatives-choice is documented.
"""
import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, cohen_kappa_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT = "{REPO_ROOT}"
RESULTS = f"{PROJECT}/results/phase3_v2"
V1_RESULTS = f"{PROJECT}/results/phase3"

DEV = ["A", "B", "C"]
VAL = ["D", "E"]
ALL = DEV + VAL

AUDIT_FEATURES = [
    "ngram13_pile", "ngram13_fineweb",
    "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
    "loss_ft", "loss_base", "refmodel_delta",
]


def load_scores(root):
    out = {}
    for v in ALL:
        p = f"{root}/{v}/scores.pkl"
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            out[v] = pickle.load(f)
    return out


def best_threshold(s, y):
    thrs = np.percentile(s, np.linspace(0, 100, 501))
    best = (None, -1.0, 0, 0)
    for t in thrs:
        pred = (s >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if f1 > best[1]:
            best = (float(t), float(f1), float(p), float(r))
    return best


def per_method_calibration(all_scores):
    methods = [k for k in next(iter(all_scores.values())).keys() if k != "label"]
    out = {}
    for m in methods:
        dev_s, dev_l, val_s, val_l = [], [], [], []
        for v, s in all_scores.items():
            if m not in s:
                continue
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
        try: auc_dev = float(roc_auc_score(dev_l, dev_s))
        except Exception: auc_dev = None

        if len(val_l) and len(np.unique(val_l)) == 2:
            pred_val = (val_s >= thr).astype(int)
            p_val, r_val, f1_val, _ = precision_recall_fscore_support(
                val_l, pred_val, average="binary", zero_division=0)
            auc_val = float(roc_auc_score(val_l, val_s))
        else:
            p_val = r_val = f1_val = auc_val = None

        out[m] = {
            "threshold": thr,
            "sign": sign,
            "dev": {"f1": f1_dev, "precision": p_dev, "recall": r_dev,
                     "auc": auc_dev, "n": int(len(dev_l))},
            "val": {"f1": f1_val, "precision": p_val, "recall": r_val,
                     "auc": auc_val, "n": int(len(val_l))},
        }
        sign_str = '+' if sign > 0 else '-'
        print(f"[{m}] sign={sign_str} dev_F1={f1_dev:.3f} dev_AUC={auc_dev:.3f} "
              f"val_F1={'--' if f1_val is None else f'{f1_val:.3f}'} "
              f"val_AUC={'--' if auc_val is None else f'{auc_val:.3f}'}")
    return out


def train_ensemble(all_scores, methods):
    def stack(variants):
        X, y = [], []
        for v in variants:
            if v not in all_scores:
                continue
            s = all_scores[v]
            if not all(m in s for m in methods):
                continue
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
    print(f"ensemble: dev={X_dev.shape} val={X_val.shape}")
    if X_dev.shape[0] == 0:
        return None

    scaler = StandardScaler().fit(X_dev)
    X_dev_s = scaler.transform(X_dev)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_dev_s, y_dev)

    dev_pred_p = clf.predict_proba(X_dev_s)[:, 1]
    auc_dev = float(roc_auc_score(y_dev, dev_pred_p))
    pred_dev = (dev_pred_p >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_dev, pred_dev, average="binary", zero_division=0)
    dev_metrics = {"f1": float(f1), "precision": float(p), "recall": float(r),
                    "auc": auc_dev, "n": int(len(y_dev))}

    val_metrics = None
    if len(X_val) and len(np.unique(y_val)) == 2:
        X_val_s = scaler.transform(X_val)
        val_p = clf.predict_proba(X_val_s)[:, 1]
        auc_val = float(roc_auc_score(y_val, val_p))
        pred_val = (val_p >= 0.5).astype(int)
        pp, rr, ff, _ = precision_recall_fscore_support(y_val, pred_val, average="binary", zero_division=0)
        val_metrics = {"f1": float(ff), "precision": float(pp), "recall": float(rr),
                        "auc": auc_val, "n": int(len(y_val))}

    return {"scaler": scaler, "clf": clf, "methods": methods,
            "dev_metrics": dev_metrics, "val_metrics": val_metrics}


def main():
    v2 = load_scores(RESULTS)
    if not v2:
        raise SystemExit("Run phase3_v2_detect.py for each variant first.")
    print(f"[v2] loaded variants: {list(v2.keys())}")

    calib = per_method_calibration(v2)

    ens_audit = train_ensemble(v2, AUDIT_FEATURES)
    ens_full = train_ensemble(v2, list(calib.keys()))

    # Try to load v1 calibration for comparison
    v1_cal_path = f"{V1_RESULTS}/calibration.json"
    v1_compare = None
    if os.path.exists(v1_cal_path):
        with open(v1_cal_path) as f:
            v1 = json.load(f)
        v1_compare = {
            "per_method_v1": v1.get("per_method", {}),
            "ensemble_audit_v1": v1.get("ensemble_audit_val") or v1.get("ensemble_val"),
            "ensemble_full_v1": v1.get("ensemble_full_val"),
        }

    out = {
        "calibration_scheme": "same_benchmark_negatives",
        "per_method": calib,
        "ensemble_audit_dev": ens_audit["dev_metrics"] if ens_audit else None,
        "ensemble_audit_val": ens_audit["val_metrics"] if ens_audit else None,
        "ensemble_full_dev": ens_full["dev_metrics"] if ens_full else None,
        "ensemble_full_val": ens_full["val_metrics"] if ens_full else None,
        "audit_features": AUDIT_FEATURES,
        "v1_comparison": v1_compare,
    }

    with open(f"{RESULTS}/calibration.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    if ens_audit:
        with open(f"{RESULTS}/ensemble.pkl", "wb") as f:
            pickle.dump({"scaler": ens_audit["scaler"], "clf": ens_audit["clf"],
                          "methods": ens_audit["methods"],
                          "signs": {m: calib[m]["sign"] for m in ens_audit["methods"]
                                     if m in calib}}, f)
    if ens_full:
        with open(f"{RESULTS}/ensemble_full.pkl", "wb") as f:
            pickle.dump({"scaler": ens_full["scaler"], "clf": ens_full["clf"],
                          "methods": ens_full["methods"],
                          "signs": {m: calib[m]["sign"] for m in ens_full["methods"]
                                     if m in calib}}, f)

    print("\n=== v1 vs v2 summary ===")
    if v1_compare and ens_audit:
        v1m = v1_compare.get("ensemble_audit_v1") or {}
        v2m = ens_audit["val_metrics"] or {}
        print(f"  ensemble val F1:  v1={v1m.get('f1','--')}  v2={v2m.get('f1','--')}")
        print(f"  ensemble val AUC: v1={v1m.get('auc','--')}  v2={v2m.get('auc','--')}")
    print(f"\n[SAVED] {RESULTS}/calibration.json, ensemble.pkl, ensemble_full.pkl")


if __name__ == "__main__":
    main()
