#!/usr/bin/env python3
"""Q5: AUC of v2 logistic-regression ensemble on calibration variant D.

Min-K%++ standalone collapses to AUC 0.18 on variant D (mixed-benchmark
fine-tune, the failure mode appendix). Does the v2 ensemble also
collapse?
"""
import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

PROJECT = "{REPO_ROOT}"
PHASE3_V2 = f"{PROJECT}/results/phase3_v2"
ENSEMBLE_V2 = f"{PROJECT}/results/phase3_v2/ensemble.pkl"
V2_FEATURES = ["ngram13_pile", "ngram13_fineweb",
                "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
                "loss_ft", "loss_base", "refmodel_delta"]


def main():
    ens = pickle.load(open(ENSEMBLE_V2, "rb"))
    scaler = ens["scaler"]; clf = ens["clf"]
    print("v2 calibration AUC and F1 per variant:")
    print(f"{'variant':<10s} {'n':>5s} {'auc':>6s} {'f1':>6s} {'acc':>6s}")
    for v in ["A", "B", "C", "D", "E"]:
        sp = f"{PHASE3_V2}/{v}/scores.pkl"
        if not os.path.exists(sp):
            continue
        d = pickle.load(open(sp, "rb"))
        y = d["label"]
        X = np.column_stack([d[f] for f in V2_FEATURES])
        X = np.nan_to_num(X)
        prob = clf.predict_proba(scaler.transform(X))[:, 1]
        auc = roc_auc_score(y, prob)
        prec, rec, _ = precision_recall_curve(y, prob)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        acc = ((prob >= 0.5) == y).mean()
        label = v
        if v == "A": label += " (GSM8K)"
        elif v == "B": label += " (MMLU)"
        elif v == "C": label += " (HSwag)"
        elif v == "D": label += " (mixed)"
        elif v == "E": label += " (Wiki)"
        print(f"{label:<12s} {len(y):>5d} {auc:>.4f} {np.nanmax(f1):>.4f} {acc:>.4f}")


if __name__ == "__main__":
    main()
