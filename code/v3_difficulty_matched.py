#!/usr/bin/env python3
"""v3: difficulty-matched calibration.

For each variant (A-E), the negatives under v3 are NOT random held-out
items from the same benchmark. Instead, for each positive item p, we pick
the negative n whose predicted difficulty (Pythia-1B per-item loss) is
closest to p's predicted difficulty. This forces the calibration to
decouple the memorization signal (f_mem) from the difficulty signal
(f_diff) that v2 still leaks.

Additionally, we add a difficulty-control feature: predicted-difficulty
quantile within the benchmark. The ensemble learns to discount this
channel at calibration time, which should prevent it from being used as
a shortcut at deployment.

We reuse the cached phase3_v2 features (same positives/negatives pool)
and simply rebalance the pos/neg pairing.
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
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
OUT = f"{PROJECT}/results/phase3_v3"
Path(OUT).mkdir(parents=True, exist_ok=True)

VARIANTS = ["A", "B", "C", "D", "E"]
DEV = ["A", "B", "C"]
VAL = ["D", "E"]

# v3 features: same 10 as v2 plus a difficulty-quantile control.
AUDIT_FEATURES = ["ngram13_pile", "ngram13_fineweb",
                   "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
                   "loss_ft", "loss_base", "refmodel_delta"]


def load_v2_calib():
    """Load cached v2 calibration scores per variant."""
    out = {}
    for v in VARIANTS:
        p = f"{PHASE3_V2}/{v}/scores.pkl"
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            out[v] = pickle.load(f)
    return out


def difficulty_match_indices(pos_diff, neg_diff):
    """For each positive, pick the negative with closest predicted-difficulty.
    Returns indices into neg_diff array (one per positive). Nearest-neighbor
    matching with replacement to keep it simple."""
    matched = np.zeros(len(pos_diff), dtype=int)
    for i, pd in enumerate(pos_diff):
        matched[i] = int(np.argmin(np.abs(neg_diff - pd)))
    return matched


def build_v3_features(variant_scores):
    """Rebuild feature matrix for v3 with difficulty-matched negatives."""
    y = variant_scores["label"]
    pos_mask = y == 1; neg_mask = y == 0
    loss_base = variant_scores["loss_base"]
    # Use loss_base (Pythia-1B loss) as predicted-difficulty proxy
    pos_diff = loss_base[pos_mask]
    neg_diff = loss_base[neg_mask]
    # Match each positive to the closest-difficulty negative (with replacement)
    matched = difficulty_match_indices(pos_diff, neg_diff)
    # Build X_v3 matrix: positives plus matched negatives
    neg_indices = np.where(neg_mask)[0]
    selected_neg = neg_indices[matched]
    pos_indices = np.where(pos_mask)[0]
    keep = np.concatenate([pos_indices, selected_neg])
    new_y = np.concatenate([np.ones(len(pos_indices)),
                             np.zeros(len(selected_neg))]).astype(int)
    # Also add the difficulty-quantile feature (rank within benchmark)
    diff_quantile = np.argsort(np.argsort(loss_base)) / max(len(loss_base) - 1, 1)
    X = np.column_stack([variant_scores[f][keep] for f in AUDIT_FEATURES]
                          + [diff_quantile[keep]])
    return X, new_y


def train_v3_ensemble(data):
    X_dev, y_dev, X_val, y_val = [], [], [], []
    for v, d in data.items():
        X, y = build_v3_features(d)
        if v in DEV:
            X_dev.append(X); y_dev.append(y)
        else:
            X_val.append(X); y_val.append(y)
    X_dev = np.vstack(X_dev); y_dev = np.concatenate(y_dev)
    X_val = np.vstack(X_val); y_val = np.concatenate(y_val)
    X_dev = np.nan_to_num(X_dev); X_val = np.nan_to_num(X_val)
    scaler = StandardScaler().fit(X_dev)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        scaler.transform(X_dev), y_dev)
    s_val = clf.predict_proba(scaler.transform(X_val))[:, 1]
    p, r, _ = precision_recall_curve(y_val, s_val)
    f1 = 2 * p * r / (p + r + 1e-12)
    return clf, scaler, {"f1_val": float(np.nanmax(f1)),
                          "auc_val": float(roc_auc_score(y_val, s_val)),
                          "n_dev": int(len(y_dev)),
                          "n_val": int(len(y_val))}


def apply_to_audit(clf, scaler):
    """Apply v3 ensemble to 22 audit models' features + difficulty-quantile."""
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
        if not all(f in feats for f in AUDIT_FEATURES):
            continue
        # Compute difficulty-quantile per benchmark (rank by loss_base within
        # each benchmark's subset of items)
        prov = d["provenance"]
        loss_base = feats["loss_base"]
        diff_q = np.zeros_like(loss_base, dtype=float)
        benches = sorted(set(b for b, _ in prov))
        for b in benches:
            idx = [i for i, (bb, _) in enumerate(prov) if bb == b]
            sub = loss_base[idx]
            ranks = np.argsort(np.argsort(sub))
            diff_q[idx] = ranks / max(len(idx) - 1, 1)
        X = np.column_stack([feats[f] for f in AUDIT_FEATURES] + [diff_q])
        X = np.nan_to_num(X)
        p_contam = clf.predict_proba(scaler.transform(X))[:, 1]
        results[model] = {"provenance": prov, "p_contam": p_contam}
    return results


def main():
    data = load_v2_calib()
    print(f"Loaded {len(data)} variants")
    clf, scaler, metrics = train_v3_ensemble(data)
    print(f"v3 ensemble: F1_val={metrics['f1_val']:.4f} AUC={metrics['auc_val']:.4f}")

    print("Applying v3 to 22-model audit...")
    per_model = apply_to_audit(clf, scaler)
    print(f"Scored {len(per_model)} models")

    # Save per-model p_contam so downstream stratified tests can pick them up
    for model, r in per_model.items():
        out_dir = f"{PROJECT}/results/phase4_v3/{model}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{out_dir}/scores.pkl", "wb") as f:
            pickle.dump(r, f)
    print(f"Saved v3 per-model scores to {PROJECT}/results/phase4_v3/")

    # Quick MATH-500 summary: does v3 reduce the residual wrong-direction bias?
    import pandas as pd
    try:
        with open(f"{PHASE5_FULL}/meta-llama__Llama-3.1-8B/per_item.pkl", "rb") as f:
            pass
    except Exception:
        pass

    # Count high-risk rates on math500 as a quick indicator
    print("\nv3 high-risk flag rates on MATH-500 (and comparison to v2):")
    from collections import defaultdict
    v3_rates = {}
    for model, r in per_model.items():
        prov = r["provenance"]; p = r["p_contam"]
        idx = [i for i, (b, _) in enumerate(prov) if b == "math500"]
        if not idx:
            continue
        sub = p[idx]
        v3_rates[model] = (float(np.mean(sub > 0.5)), float(np.mean(sub)))
    for m, (hr, mean) in sorted(v3_rates.items()):
        print(f"  {m:<42s}  >=0.5: {hr:.3f}  mean: {mean:.3f}")


if __name__ == "__main__":
    main()
