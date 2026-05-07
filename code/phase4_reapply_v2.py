#!/usr/bin/env python3
"""Re-apply the v2 ensemble (trained on same-benchmark negatives) to the
existing Phase 4 per-model feature matrices, producing results/phase4_v2/.

Reuses ~113K forward-pass computations from the original Phase 4 audit.
Only the ensemble prediction step changes, so this is cheap.
"""
import json
import os
import pickle
import shutil
import time

import numpy as np

PROJECT = "{REPO_ROOT}"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
ENSEMBLE_V2 = f"{PROJECT}/results/phase3_v2/ensemble.pkl"


def main():
    if not os.path.exists(ENSEMBLE_V2):
        raise SystemExit(f"Missing v2 ensemble: {ENSEMBLE_V2}")
    with open(ENSEMBLE_V2, "rb") as f:
        ens = pickle.load(f)
    methods = ens["methods"]
    print(f"v2 ensemble methods: {methods}")

    os.makedirs(PHASE4_V2, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    for model_dir in sorted(os.listdir(PHASE4_V1)):
        if model_dir.startswith("_"):
            continue
        v1_scores = f"{PHASE4_V1}/{model_dir}/scores.pkl"
        v1_summary = f"{PHASE4_V1}/{model_dir}/summary.json"

        if not os.path.exists(v1_scores):
            if os.path.exists(v1_summary):
                with open(v1_summary) as f:
                    s = json.load(f)
                if s.get("_status") == "failed":
                    # Propagate the failure marker
                    out_dir = f"{PHASE4_V2}/{model_dir}"
                    os.makedirs(out_dir, exist_ok=True)
                    with open(f"{out_dir}/summary.json", "w") as f:
                        json.dump(s, f, indent=2)
                    print(f"[SKIP] {model_dir}: propagated fail marker")
            continue

        with open(v1_scores, "rb") as f:
            data = pickle.load(f)
        features = dict(data["features"])
        if "loss_audit" in features and "loss_ft" not in features:
            features["loss_ft"] = features.pop("loss_audit")
        if "loss_ref" in features and "loss_base" not in features:
            features["loss_base"] = features.pop("loss_ref")

        n_rows = len(next(iter(features.values())))
        cols, missing = [], []
        for m in methods:
            if m in features:
                cols.append(np.asarray(features[m], dtype=np.float64))
            else:
                missing.append(m)
                cols.append(np.full(n_rows, np.nan, dtype=np.float64))
        if missing:
            print(f"[WARN] {model_dir}: missing {missing}")
        X = np.stack(cols, axis=1)
        for j in range(X.shape[1]):
            col = X[:, j]
            nan = np.isnan(col)
            if nan.any():
                col[nan] = ens["scaler"].mean_[j]
        X_s = ens["scaler"].transform(X)
        p_contam = ens["clf"].predict_proba(X_s)[:, 1].astype(np.float32)

        out_dir = f"{PHASE4_V2}/{model_dir}"
        os.makedirs(out_dir, exist_ok=True)
        new_data = dict(data)
        new_data["p_contam"] = p_contam
        new_data["features"] = {k: np.asarray(v, dtype=np.float32)
                                 for k, v in features.items()}
        new_data["_ensemble_version"] = "v2_same_bench_negatives"
        new_data["_created_ts"] = ts
        with open(f"{out_dir}/scores.pkl", "wb") as f:
            pickle.dump(new_data, f)

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
        with open(f"{out_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        mean_p = float(np.nanmean(p_contam))
        hr = float((p_contam > 0.5).mean())
        print(f"[OK] {model_dir}: v2 p_mean={mean_p:.3f} high_risk={hr:.3f}")


if __name__ == "__main__":
    main()
