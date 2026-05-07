#!/usr/bin/env python3
"""Baseline comparison: Min-K% Probability standalone detector vs v2 ensemble.

Reviewer concern: the paper compares v1 vs v2 but not v2 against
existing literature detectors. We address this by running a Min-K%
standalone detector on the same 22-model x 13-benchmark audit matrix
and computing the same stratified clean-minus-contaminated accuracy
gaps. Min-K% uses the mean of the lowest k% per-token log-probabilities
per item; we use the mink_5 feature already cached in phase4. Threshold
is the within-benchmark median (the simplest standalone deployment),
mirroring how a practitioner would use a standalone detector without
ensemble training.

Output:
  results/baseline_minkpp/stats.csv                 # per-cell stats
  results/baseline_minkpp/summary.json              # aggregate comparison vs v2
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = "{REPO_ROOT}"
PHASE4 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5 = f"{PROJECT}/results/phase5_full"
OUT = f"{PROJECT}/results/baseline_minkpp"
Path(OUT).mkdir(parents=True, exist_ok=True)

BENCHES = ["gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge",
            "humaneval", "mbpp", "truthfulqa_mc"]

UNIFORM = {"gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge",
            "humaneval", "mbpp"}


def perm_test(cc, ct, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = cc.mean() - ct.mean()
    combined = np.concatenate([cc, ct])
    extreme = 0
    n_cc = len(cc)
    for _ in range(n_perm):
        rng.shuffle(combined)
        c = combined[:n_cc].mean() - combined[n_cc:].mean()
        if c <= obs:
            extreme += 1
    return float(obs), extreme / n_perm


def bootstrap_ci(cc, ct, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(cc, size=len(cc), replace=True).mean()
        b = rng.choice(ct, size=len(ct), replace=True).mean()
        gaps[i] = a - b
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def load_correct(model, bench):
    path = f"{PHASE5}/{model}/per_item.pkl"
    if not os.path.exists(path):
        return None
    d = pickle.load(open(path, "rb"))
    if bench not in d:
        return None
    items = d[bench]["per_item"]
    return np.array([it["correct"] for it in items], dtype=float)


def load_feat_and_prov(model, bench, which="v1"):
    base = PHASE4 if which == "v1" else PHASE4_V2
    sp = f"{base}/{model}/scores.pkl"
    if not os.path.exists(sp):
        return None
    d = pickle.load(open(sp, "rb"))
    idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench]
    if not idxs:
        return None
    item_ids = np.array([d["provenance"][i][1] for i in idxs])
    feats = {k: v[idxs] for k, v in d["features"].items()}
    p_contam = d["p_contam"][idxs] if "p_contam" in d else None
    order = np.argsort(item_ids)
    feats = {k: v[order] for k, v in feats.items()}
    if p_contam is not None:
        p_contam = p_contam[order]
    item_ids = item_ids[order]
    return feats, p_contam, item_ids


def cell_stat_standalone(correct, score, sign="high_is_contam", min_n=10):
    """Threshold score at median; HIGH side = contaminated."""
    thr = np.median(score)
    if sign == "high_is_contam":
        contam = score >= thr
    else:
        contam = score <= thr
    clean = ~contam
    nc = int(clean.sum()); nt = int(contam.sum())
    if nc < min_n or nt < min_n:
        return None
    cc = correct[clean]; ct = correct[contam]
    gap, p = perm_test(cc, ct)
    lo, hi = bootstrap_ci(cc, ct)
    return dict(n_clean=nc, n_contam=nt, clean_acc=float(cc.mean()),
                contam_acc=float(ct.mean()), obs_gap=float(gap),
                ci_low=lo, ci_high=hi, perm_p=float(p))


def cell_stat_ensemble(correct, p_contam, min_n=10):
    clean = p_contam < 0.1
    contam = p_contam > 0.5
    nc = int(clean.sum()); nt = int(contam.sum())
    if nc < min_n or nt < min_n:
        return None
    cc = correct[clean]; ct = correct[contam]
    gap, p = perm_test(cc, ct)
    lo, hi = bootstrap_ci(cc, ct)
    return dict(n_clean=nc, n_contam=nt, clean_acc=float(cc.mean()),
                contam_acc=float(ct.mean()), obs_gap=float(gap),
                ci_low=lo, ci_high=hi, perm_p=float(p))


def main():
    models = sorted([m for m in os.listdir(PHASE4) if not m.startswith("_")])
    print(f"Auditing {len(models)} models x {len(BENCHES)} benchmarks")

    rows = []
    for model in models:
        for bench in BENCHES:
            correct = load_correct(model, bench)
            if correct is None:
                continue
            # v1-side features contain mink_{5,10,20,30,50}
            fp = load_feat_and_prov(model, bench, which="v1")
            if fp is None:
                continue
            feats, _, _ = fp
            n = len(correct)
            if len(feats["mink_5"]) != n:
                m = min(n, len(feats["mink_5"]))
                correct = correct[:m]
                feats = {k: v[:m] for k, v in feats.items()}

            for k_ in [5, 10, 20, 30, 50]:
                score = feats[f"mink_{k_}"]
                # Min-K%: low mean log-prob of lowest-k% = confident = memorized.
                # mink feature stored as negative log-prob, so HIGH values = likely memorized.
                r = cell_stat_standalone(correct, score, sign="high_is_contam")
                if r is None:
                    continue
                rows.append(dict(model=model, benchmark=bench,
                                  detector=f"mink_{k_}", **r))

            # v2 ensemble
            fp2 = load_feat_and_prov(model, bench, which="v2")
            if fp2 is None:
                continue
            _, p_contam, _ = fp2
            if p_contam is None or len(p_contam) < n:
                continue
            p_contam = p_contam[:len(correct)]
            r = cell_stat_ensemble(correct, p_contam)
            if r is not None:
                rows.append(dict(model=model, benchmark=bench,
                                  detector="v2_ensemble", **r))

    # Save per-cell CSV
    import csv
    fieldnames = list(rows[0].keys())
    with open(f"{OUT}/stats.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} cell rows to {OUT}/stats.csv")

    # Aggregate summary
    summary = {}
    for det in ["mink_5", "mink_10", "mink_20", "mink_30", "mink_50", "v2_ensemble"]:
        sub = [r for r in rows if r["detector"] == det
                and r["benchmark"] in UNIFORM]
        if not sub:
            continue
        gaps = np.array([r["obs_gap"] for r in sub])
        ps = np.array([r["perm_p"] for r in sub])
        summary[det] = {
            "n_cells_uniform": len(sub),
            "mean_gap": float(gaps.mean()),
            "median_gap": float(np.median(gaps)),
            "n_sig_correct_raw": int((ps < 0.05).sum()),
            "n_sig_wrong_raw": int((ps > 0.95).sum()),
        }
        # Bonferroni correction at m = # cells
        m = len(sub)
        summary[det]["n_sig_correct_bonf"] = int((ps < 0.05 / m).sum())
        summary[det]["n_sig_wrong_bonf"] = int((ps > 1 - 0.05 / m).sum())

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
