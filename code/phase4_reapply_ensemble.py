#!/usr/bin/env python3
"""Re-apply the calibrated ensemble to Phase 4 scores.pkl files.

Fixes two bugs in the original Phase 4 run:
  1. Feature names were `loss_audit`/`loss_ref` but ensemble expects
     `loss_ft`/`loss_base` — those features were imputed with scaler means.
  2. Sign-flipping was applied before feeding to the ensemble, but the
     ensemble was trained on raw (unflipped) scores — double-negated mink
     contributions → saturation.

Walks every model dir under results/phase4/, rewrites p_contam and summary.json.
Backs up old files with `.bak.{timestamp}` suffix (never deletes).
"""
import json
import os
import pickle
import shutil
import time

import numpy as np

PROJECT = "{REPO_ROOT}"
PHASE4 = f"{PROJECT}/results/phase4"
ENSEMBLE = f"{PROJECT}/results/phase3/ensemble.pkl"


def main():
    with open(ENSEMBLE, "rb") as f:
        ens = pickle.load(f)
    methods = ens["methods"]
    print(f"Ensemble methods: {methods}")

    ts = time.strftime("%Y%m%d_%H%M%S")

    for model_dir in sorted(os.listdir(PHASE4)):
        if model_dir.startswith("_"):
            continue
        mp = f"{PHASE4}/{model_dir}"
        scores_path = f"{mp}/scores.pkl"
        summary_path = f"{mp}/summary.json"
        if not os.path.exists(scores_path):
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    s = json.load(f)
                if s.get("_status") == "failed":
                    print(f"[SKIP] {model_dir}: marked failed")
                    continue
            print(f"[SKIP] {model_dir}: no scores.pkl")
            continue

        shutil.copy(scores_path, f"{scores_path}.bak.{ts}")
        if os.path.exists(summary_path):
            shutil.copy(summary_path, f"{summary_path}.bak.{ts}")

        with open(scores_path, "rb") as f:
            data = pickle.load(f)
        features = dict(data["features"])
        if "loss_audit" in features and "loss_ft" not in features:
            features["loss_ft"] = features.pop("loss_audit")
        if "loss_ref" in features and "loss_base" not in features:
            features["loss_base"] = features.pop("loss_ref")

        cols, missing = [], []
        n_rows = len(next(iter(features.values())))
        for m in methods:
            if m in features:
                cols.append(np.asarray(features[m], dtype=np.float64))
            else:
                missing.append(m)
                cols.append(np.full(n_rows, np.nan, dtype=np.float64))
        if missing:
            print(f"[WARN] {model_dir}: missing features {missing}")
        X = np.stack(cols, axis=1)
        for j in range(X.shape[1]):
            col = X[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col[nan_mask] = ens["scaler"].mean_[j]
        X_s = ens["scaler"].transform(X)
        p_contam = ens["clf"].predict_proba(X_s)[:, 1].astype(np.float32)

        data["p_contam"] = p_contam
        data["features"] = {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}
        data["_reensemble_ts"] = ts
        with open(scores_path, "wb") as f:
            pickle.dump(data, f)

        prov = data["provenance"]
        loss_audit = features.get("loss_ft", np.full(n_rows, np.nan))
        refmodel_delta = features.get("refmodel_delta", np.full(n_rows, np.nan))
        benchmarks_present = sorted(set(p[0] for p in prov))
        summary = {}
        for b in benchmarks_present:
            mask = np.array([pp[0] == b for pp in prov])
            if not mask.any():
                continue
            summary[b] = {
                "n_items": int(mask.sum()),
                "p_contam_mean": float(np.nanmean(p_contam[mask])),
                "p_contam_p90": float(np.nanpercentile(p_contam[mask], 90)),
                "high_risk_rate": float((p_contam[mask] > 0.5).mean()),
                "mean_loss_audit": float(np.nanmean(loss_audit[mask])),
                "mean_refmodel_delta": float(np.nanmean(refmodel_delta[mask])),
            }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        mean_p = float(np.nanmean(p_contam))
        p90 = float(np.nanpercentile(p_contam, 90))
        hr = float((p_contam > 0.5).mean())
        print(f"[OK] {model_dir}: p_contam mean={mean_p:.3f} p90={p90:.3f} high_risk={hr:.3f}")


if __name__ == "__main__":
    main()
