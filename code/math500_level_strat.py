#!/usr/bin/env python3
"""Stage M: MATH-500 level-stratified impact analysis.

R1/R4/R5 all flagged "residual difficulty confound on variant-difficulty
benchmarks". MATH-500 has a native `level` field (1-5). Under length-quartile
stratification v2 fails. Does level-stratification resolve it?
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
MATH_EVAL = f"{PROJECT}/results/math_eval"
OUT = f"{PROJECT}/results/math500_level_strat"
Path(OUT).mkdir(parents=True, exist_ok=True)


def load_math500():
    items = []
    with open(f"{BENCH_DIR}/math500.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    items = load_math500()
    print(f"MATH-500: {len(items)} items")
    # Check level distribution
    from collections import Counter
    levels = [it.get("level") for it in items]
    print(f"Level distribution: {Counter(levels)}")
    # Normalize level field
    def norm_level(it):
        lvl = it.get("level", "")
        if isinstance(lvl, int): return lvl
        if isinstance(lvl, str):
            import re
            m = re.search(r"\d", lvl)
            return int(m.group()) if m else 0
        return 0
    bins = np.array([norm_level(it) for it in items])
    print(f"Level-bin distribution: {Counter(bins.tolist())}")

    rows = []
    for model in sorted(os.listdir(PHASE4_V2)):
        if model.startswith("_"): continue
        sp = f"{PHASE4_V2}/{model}/scores.pkl"
        if not os.path.exists(sp): continue
        try:
            with open(sp, "rb") as f: d = pickle.load(f)
        except Exception: continue
        idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == "math500"]
        if not idxs: continue
        item_ids = [d["provenance"][i][1] for i in idxs]
        p = d["p_contam"][idxs]
        order = np.argsort(item_ids)
        p = p[order][:len(items)]

        # Load accuracy from math_eval
        ma = f"{MATH_EVAL}/{model}_math500/results.pkl"
        if not os.path.exists(ma): continue
        with open(ma, "rb") as f: results = pickle.load(f)
        correct = np.array([r["correct"] for r in results], dtype=np.int32)

        n = min(len(correct), len(p), len(bins))
        correct = correct[:n]; pp = p[:n]; bb = bins[:n]
        # Level-stratified
        gaps, ses = [], []
        for b in sorted(np.unique(bb)):
            m = bb == b
            cc = correct[m & (pp < 0.1)]; ct = correct[m & (pp > 0.5)]
            if len(cc) < 5 or len(ct) < 5: continue
            gap = cc.mean() - ct.mean()
            se = np.sqrt(cc.var(ddof=1)/len(cc) + ct.var(ddof=1)/len(ct))
            if se < 1e-6 or not np.isfinite(se): continue
            gaps.append(gap); ses.append(se)
        if not gaps: continue
        gaps = np.array(gaps); ses = np.array(ses)
        w = 1/ses**2
        iv_gap = float((w*gaps).sum()/w.sum())
        iv_se = float(1/np.sqrt(w.sum()))
        z = iv_gap/iv_se
        p_correct = float(stats.norm.cdf(z))
        rows.append({"model": model, "n_strata": len(gaps),
                     "iv_gap": iv_gap, "iv_se": iv_se, "z": z,
                     "p_correct": p_correct, "p_two": 2*min(p_correct,1-p_correct)})

    df = pd.DataFrame(rows).sort_values("p_correct")
    df.to_csv(f"{OUT}/math500_level_strat.csv", index=False)

    # Compute Bonferroni survival
    m = len(df)
    bonf = 0.05/m if m>0 else 0.05
    n_raw = int(((df.iv_gap<0)&(df.p_correct<0.05)).sum())
    n_bonf = int(((df.iv_gap<0)&(df.p_correct<bonf)).sum())
    # Holm / BH on correct-direction only
    corr_mask = df.iv_gap<0
    if corr_mask.any():
        pv = df.loc[corr_mask, "p_two"].values
        rej_holm, _, _, _ = multipletests(pv, alpha=0.05, method="holm")
        rej_bh, _, _, _ = multipletests(pv, alpha=0.05, method="fdr_bh")
        n_holm = int(rej_holm.sum()); n_bh = int(rej_bh.sum())
    else:
        n_holm = 0; n_bh = 0
    print(f"\n=== MATH-500 v2 level-stratified results (m={m}) ===")
    print(f"raw_sig_correct={n_raw} bonf={n_bonf} holm={n_holm} bh={n_bh}")
    print("\nAll cells:")
    for _, r in df.iterrows():
        flag = "*" if r["p_correct"]<0.05 and r["iv_gap"]<0 else " "
        print(f" {flag} {r.model:<42s} strata={int(r.n_strata)} iv_gap={r.iv_gap:+.4f} p_corr={r.p_correct:.4f}")


if __name__ == "__main__":
    main()
