#!/usr/bin/env python3
"""GSM8K step-count stratifier (principled difficulty proxy).

Number of arithmetic steps = number of <<...>> markers in the gold answer.
This is a more defensible difficulty proxy than character length since it
directly measures reasoning depth. Test whether OLMo-2-7B × GSM8K cell
survives Bonferroni under step-count stratification.
"""
import json
import os
import pickle
import re
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
OUT = f"{PROJECT}/results/stepcount_strat"
Path(OUT).mkdir(parents=True, exist_ok=True)


def step_count(answer: str) -> int:
    """Count arithmetic steps as number of <<...>> markers."""
    return len(re.findall(r"<<[^>]*>>", answer or ""))


def bin_steps(n_steps: int) -> int:
    if n_steps <= 2: return 0
    if n_steps <= 4: return 1
    if n_steps <= 6: return 2
    return 3


def main():
    # Load GSM8K
    items = []
    with open(f"{BENCH_DIR}/gsm8k_test.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    step_bins = np.array([bin_steps(step_count(it.get("answer", ""))) for it in items])
    n_items = len(items)
    print(f"GSM8K items: {n_items}")
    print("Step-count bin distribution:",
          {int(b): int((step_bins == b).sum()) for b in range(4)})

    # For each model, compute stratified gap under step-count stratification
    rows = []
    for model_short in sorted(os.listdir(PHASE4_V2)):
        if model_short.startswith("_"):
            continue
        sp = f"{PHASE4_V2}/{model_short}/scores.pkl"
        if not os.path.exists(sp):
            continue
        try:
            with open(sp, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == "gsm8k_test"]
        if not idxs:
            continue
        item_ids = [d["provenance"][i][1] for i in idxs]
        p = d["p_contam"][idxs]
        order = np.argsort(item_ids)
        p = p[order][:n_items]
        # Load phase5 accuracy
        pi_path = f"{PHASE5_FULL}/{model_short}/per_item.pkl"
        if not os.path.exists(pi_path):
            continue
        with open(pi_path, "rb") as f:
            pi = pickle.load(f)
        if "gsm8k_test" not in pi:
            continue
        correct = np.array([r["correct"] for r in pi["gsm8k_test"]["per_item"]],
                           dtype=np.int32)
        n = min(len(correct), len(p), len(step_bins))
        correct = correct[:n]; p = p[:n]; bins = step_bins[:n]

        # Stratified gap
        gaps, ses = [], []
        for b in range(4):
            m = bins == b
            if m.sum() < 5:
                continue
            cc = correct[m & (p < 0.1)]
            ct = correct[m & (p > 0.5)]
            if len(cc) < 5 or len(ct) < 5:
                continue
            gap = cc.mean() - ct.mean()
            se = np.sqrt(cc.var(ddof=1) / len(cc) + ct.var(ddof=1) / len(ct))
            if se < 1e-6 or not np.isfinite(se):
                continue
            gaps.append(gap); ses.append(se)
        if len(gaps) < 1:
            continue
        gaps = np.array(gaps); ses = np.array(ses)
        w = 1 / ses**2
        iv_gap = float((w * gaps).sum() / w.sum())
        iv_se = float(1 / np.sqrt(w.sum()))
        z = iv_gap / iv_se
        p_correct = float(stats.norm.cdf(z))
        p_two = float(2 * min(p_correct, 1 - p_correct))
        rows.append({"model": model_short.replace("__", "/"),
                     "n_strata": len(gaps), "iv_gap": iv_gap, "iv_se": iv_se,
                     "z": z, "p_correct": p_correct, "p_two": p_two})

    # Sort by p_correct
    rows.sort(key=lambda r: r["p_correct"])
    print("\n=== GSM8K step-count-stratified v2 cells ===")
    for r in rows:
        flag = "***" if r["p_correct"] < 0.05 and r["iv_gap"] < 0 else "   "
        print(f"{flag} {r['model']:<30s} strata={r['n_strata']} "
              f"iv_gap={r['iv_gap']:+.4f} ± {r['iv_se']:.4f}  "
              f"z={r['z']:+.3f}  p_corr={r['p_correct']:.4f}  p_two={r['p_two']:.4f}")

    # Save
    import pandas as pd
    pd.DataFrame(rows).to_csv(f"{OUT}/gsm8k_stepcount_v2.csv", index=False)
    print(f"\nSaved to {OUT}/gsm8k_stepcount_v2.csv")


if __name__ == "__main__":
    main()
