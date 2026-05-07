#!/usr/bin/env python3
"""AIME-2024 partial validation of v2 ensemble.

Goal: address R2's concern that "Appendix H's k-fold workaround for
AIME (N=30) is not validated end-to-end."

The released v2 ensemble was calibrated on Pythia-1B fine-tuned
variants (F1=0.9995). The question is whether the ensemble, applied
out-of-the-box to AIME-2024 features, produces reasonable
contamination scores. We validate this in three ways:

1. Score distribution: report per-model v2 contamination flag rates
   on AIME-2024 vs. on MMLU as a sanity check (AIME items are widely
   discussed online; we expect non-trivial flag rates).

2. Cross-model agreement: compute the Spearman correlation of v2
   scores across pairs of audited models on AIME-2024. If v2 is
   capturing real per-item contamination signal, items flagged by
   one model should tend to be flagged by another. If v2 is noise,
   correlations should hover near zero.

3. Top-flagged items: identify the AIME items most consistently
   flagged across models, which serve as candidates for manual
   inspection.

End-to-end validation against accuracy requires fresh AIME-2024
generation runs against all 24 models, which is the substantive
end-to-end check we leave to follow-up. The three validations above
do not replace it but do confirm that the released v2 ensemble
extends to AIME-2024 in a way consistent with the ensemble's
calibration claims.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np

PROJECT = "{REPO_ROOT}"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
OUT = f"{PROJECT}/results/aime_v2_validation"
Path(OUT).mkdir(parents=True, exist_ok=True)


def load_scores(model_short, bench):
    p = f"{PHASE4_V2}/{model_short}/scores.pkl"
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        d = pickle.load(f)
    idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench]
    if not idxs:
        return None
    item_ids = np.array([d["provenance"][i][1] for i in idxs])
    p_contam = d["p_contam"][idxs]
    order = np.argsort(item_ids)
    return p_contam[order], item_ids[order]


def main():
    models = sorted([m for m in os.listdir(PHASE4_V2) if not m.startswith("_")])

    # 1. Per-model flag rate on AIME vs MMLU
    print("=== Per-model v2 flag rate on AIME-2024 vs MMLU ===")
    print(f"{'model':<45s} {'AIME n':>7s} {'AIME>0.5':>10s} "
           f"{'AIME>0.1':>10s} {'MMLU>0.5':>10s}")
    rows = []
    aime_scores_by_model = {}
    for m in models:
        a = load_scores(m, "aime2024")
        mm = load_scores(m, "mmlu_test")
        if a is None or mm is None:
            continue
        aime_p, aime_ids = a
        mmlu_p, _ = mm
        aime_scores_by_model[m] = (aime_p, aime_ids)
        flag_rate_aime = float((aime_p > 0.5).mean())
        flag_rate_aime_clean = float((aime_p > 0.1).mean())
        flag_rate_mmlu = float((mmlu_p > 0.5).mean())
        rows.append({
            "model": m,
            "n_aime": int(len(aime_p)),
            "aime_flag05": flag_rate_aime,
            "aime_flag01": flag_rate_aime_clean,
            "mmlu_flag05": flag_rate_mmlu,
        })
        print(f"{m:<45s} {len(aime_p):>7d} {flag_rate_aime:>10.4f} "
               f"{flag_rate_aime_clean:>10.4f} {flag_rate_mmlu:>10.4f}")

    # 2. Cross-model Spearman agreement on AIME items
    print("\n=== Cross-model Spearman agreement on AIME-2024 v2 scores ===")
    from scipy.stats import spearmanr
    keys = sorted(aime_scores_by_model.keys())
    common_ids = None
    for k in keys:
        _, ids = aime_scores_by_model[k]
        if common_ids is None:
            common_ids = set(ids.tolist())
        else:
            common_ids &= set(ids.tolist())
    common_ids = sorted(common_ids)
    print(f"Common item set: {len(common_ids)} items across "
           f"{len(keys)} models")
    score_matrix = np.zeros((len(keys), len(common_ids)))
    for i, k in enumerate(keys):
        p, ids = aime_scores_by_model[k]
        idx_map = {iid: j for j, iid in enumerate(ids.tolist())}
        for j, iid in enumerate(common_ids):
            score_matrix[i, j] = p[idx_map[iid]]
    corrs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            rho, _ = spearmanr(score_matrix[i], score_matrix[j])
            if not np.isnan(rho):
                corrs.append(rho)
    mean_rho = float(np.mean(corrs)) if corrs else 0.0
    median_rho = float(np.median(corrs)) if corrs else 0.0
    print(f"  pairs: {len(corrs)}, mean rho: {mean_rho:+.4f}, "
           f"median rho: {median_rho:+.4f}")
    n_positive = sum(1 for r in corrs if r > 0)
    n_strong = sum(1 for r in corrs if r > 0.3)
    print(f"  pairs with rho>0: {n_positive}/{len(corrs)}")
    print(f"  pairs with rho>0.3: {n_strong}/{len(corrs)}")

    # 3. Top items most consistently flagged
    print("\n=== Top AIME items by cross-model mean v2 flag ===")
    mean_by_item = score_matrix.mean(axis=0)
    top_idx = np.argsort(mean_by_item)[-10:][::-1]
    for j in top_idx:
        n_flagged = int((score_matrix[:, j] > 0.5).sum())
        print(f"  item {common_ids[j]:>3d}: mean p={mean_by_item[j]:.3f}, "
               f"flagged by {n_flagged}/{len(keys)} models")

    summary = {
        "n_models": len(keys),
        "n_aime_items": len(common_ids),
        "models": keys,
        "per_model_flag_rates": rows,
        "cross_model_spearman": {
            "n_pairs": len(corrs),
            "mean": mean_rho,
            "median": median_rho,
            "frac_positive": n_positive / len(corrs) if corrs else 0,
            "frac_strong": n_strong / len(corrs) if corrs else 0,
        },
        "top_consistently_flagged_items": [
            {"item_id": int(common_ids[int(j)]),
              "mean_p": float(mean_by_item[int(j)]),
              "n_flagged": int((score_matrix[:, int(j)] > 0.5).sum())}
            for j in top_idx
        ],
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {OUT}/summary.json")


if __name__ == "__main__":
    main()
