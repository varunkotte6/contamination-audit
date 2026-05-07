#!/usr/bin/env python3
"""v8 final analysis with significance testing.

v8 = capability-matched pairwise reference detector for MATH-500.

For each audited model m, select K reference models with closest raw
MATH-500 accuracy. Define memorization score per item:
    delta_v8(m, i) = answer_loss(m, i) - mean_k(answer_loss(r_k, i)).

Low delta_v8 on item i relative to m's distribution = anomalously
confident prediction beyond what capability peers would give =
memorization signature.

Stratified gap test:
- contam subset: delta_v8 <= q_0.20 (bottom 20% within-model)
- clean subset:  delta_v8 >= q_0.80 (top 20% within-model)
- gap = acc(clean) - acc(contam). Correct direction: gap < 0.

We also run a permutation test for statistical significance.
"""
import os
import pickle
import numpy as np
from scipy import stats

PROJECT = "{REPO_ROOT}"
ANS_DIR = f"{PROJECT}/results/v6_answer_loss"
MATH_EVAL = f"{PROJECT}/results/math_eval"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
OUT = f"{PROJECT}/results/v8_final"
os.makedirs(OUT, exist_ok=True)


def load_math500_correct(slug):
    p = f"{MATH_EVAL}/{slug}_math500/results.pkl"
    if not os.path.exists(p):
        return None
    r = pickle.load(open(p, "rb"))
    return np.array([x["correct"] for x in r], dtype=float)


def load_v2_pcontam(slug):
    p = f"{PHASE4_V2}/{slug}/scores.pkl"
    if not os.path.exists(p):
        return None
    d = pickle.load(open(p, "rb"))
    idx = [i for i, (b, _) in enumerate(d["provenance"]) if b == "math500"]
    if not idx:
        return None
    item_ids = np.array([d["provenance"][i][1] for i in idx])
    order = np.argsort(item_ids)
    return d["p_contam"][idx][order]


def perm_test(cc, ct, n_perm=5000, seed=0):
    """One-sided permutation test for H1: mean(cc) < mean(ct)."""
    rng = np.random.default_rng(seed)
    obs = cc.mean() - ct.mean()
    combined = np.concatenate([cc, ct])
    n_cc = len(cc)
    n_less = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        c = combined[:n_cc].mean() - combined[n_cc:].mean()
        if c <= obs:
            n_less += 1
    return float(obs), n_less / n_perm


def model_family(slug):
    """Coarse family identifier for excluding same-family peers."""
    s = slug.lower()
    if "pythia" in s: return "pythia"
    if "qwen" in s: return "qwen"
    if "gemma" in s: return "gemma"
    if "olmo" in s: return "olmo"
    if "llama" in s: return "llama"
    if "phi" in s: return "phi"
    if "mistral" in s: return "mistral"
    if "yi-" in s or "/yi" in s: return "yi"
    if "deepseek" in s: return "deepseek"
    if "granite" in s: return "granite"
    return "other"


def within_z(x):
    m = np.nanmean(x); s = np.nanstd(x) + 1e-6
    return (x - m) / s


def main():
    slugs = sorted([f[:-4] for f in os.listdir(ANS_DIR) if f.endswith(".npz")])
    al_raw = {s: np.load(f"{ANS_DIR}/{s}.npz")["answer_loss"] for s in slugs}
    # Within-model z-score for v8d
    al = {s: within_z(al_raw[s]) for s in slugs}
    accs = {}
    for s in slugs:
        c = load_math500_correct(s)
        accs[s] = float(c.mean()) if c is not None else None

    K = 7  # max peers
    tail_q = 0.2
    max_diff = 0.15  # peers must be within ±0.15 raw accuracy of target
    print(f"=== v8 final analysis: max {K} peers within ±{max_diff} acc, "
          f"exclude same-family, tail={tail_q} ===\n")
    print(f"{'model':<45s} {'acc':>5s} {'refs':>5s} "
           f"{'n_cl':>4s} {'n_ct':>4s} "
           f"{'cl_a':>5s} {'ct_a':>5s} {'v8':>7s} {'perm_p':>8s} "
           f"{'v2':>7s}")
    v8_results = []; v2_results = []
    for m in slugs:
        if accs.get(m) is None:
            continue
        m_al = al[m]
        c = load_math500_correct(m)
        m_acc = accs[m]
        m_fam = model_family(m)
        others = [(s, abs(accs[s] - m_acc)) for s in slugs
                  if s != m and accs.get(s) is not None
                  and abs(accs[s] - m_acc) <= max_diff
                  and model_family(s) != m_fam]
        others.sort(key=lambda kv: kv[1])
        ref_slugs = [s for s, _ in others[:K]]
        if len(ref_slugs) < 3:  # require at least 3 peers
            continue
        ref_al = np.stack([al[s] for s in ref_slugs], axis=0)
        n = min(len(m_al), ref_al.shape[1], len(c))
        m_n = m_al[:n]; ref_n = ref_al[:, :n]; c_n = c[:n]
        valid = ~np.isnan(m_n) & ~np.isnan(ref_n).any(axis=0)
        if valid.sum() < 30:
            continue
        delta = m_n[valid] - ref_n[:, valid].mean(axis=0)
        cv = c_n[valid]
        tau_lo = np.quantile(delta, tail_q)
        tau_hi = np.quantile(delta, 1 - tail_q)
        contam = delta <= tau_lo
        clean = delta >= tau_hi
        if contam.sum() < 10 or clean.sum() < 10:
            continue
        cl_acc = float(cv[clean].mean())
        ct_acc = float(cv[contam].mean())
        gap = cl_acc - ct_acc
        _, perm_p = perm_test(cv[clean], cv[contam])
        v8_results.append((m, gap, perm_p, int(clean.sum()), int(contam.sum()),
                            m_acc))
        # v2 baseline
        p_v2 = load_v2_pcontam(m)
        v2s = "   nan"
        if p_v2 is not None:
            p_v2 = p_v2[:n][valid]
            v2_clean = p_v2 < 0.1; v2_contam = p_v2 > 0.5
            if v2_clean.sum() >= 10 and v2_contam.sum() >= 10:
                v2_gap = float(cv[v2_clean].mean() - cv[v2_contam].mean())
                _, v2_perm_p = perm_test(cv[v2_clean], cv[v2_contam])
                v2_results.append((m, v2_gap, v2_perm_p))
                v2s = f"{v2_gap:+.3f}"
        print(f"{m:<45s} {m_acc:>.3f} {K:>5d} "
               f"{int(clean.sum()):>4d} {int(contam.sum()):>4d} "
               f"{cl_acc:>.3f} {ct_acc:>.3f} "
               f"{gap:>+.3f} {perm_p:>8.1e} {v2s:>7s}")
    if v8_results:
        gaps = [x[1] for x in v8_results]
        ps = [x[2] for x in v8_results]
        print(f"\nv8 mean_gap = {np.mean(gaps):+.4f}")
        print(f"v8 median_gap = {np.median(gaps):+.4f}")
        print(f"v8 n_negative = {sum(1 for g in gaps if g < 0)}/{len(gaps)}")
        print(f"v8 n_perm_p<0.05 = {sum(1 for p in ps if p < 0.05)}/{len(ps)}")
    if v2_results:
        gaps = [x[1] for x in v2_results]
        ps = [x[2] for x in v2_results]
        print(f"v2 mean_gap = {np.mean(gaps):+.4f}")
        print(f"v2 n_negative = {sum(1 for g in gaps if g < 0)}/{len(gaps)}")
        print(f"v2 n_perm_p<0.05 = {sum(1 for p in ps if p < 0.05)}/{len(ps)}")
    # Paired Wilcoxon across cells
    if v8_results and v2_results:
        common = [(v8, v2) for v8, v2 in zip(v8_results, v2_results)
                   if v8[0] == v2[0]]
        if common:
            diffs = [v8[1] - v2[1] for v8, v2 in common]
            w, p = stats.wilcoxon(diffs, alternative="less")
            print(f"\nPaired Wilcoxon v8 < v2 (one-sided): "
                  f"W={w:.1f}, p={p:.4f}, n_pairs={len(common)}")
    # Save results
    import json
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"v8": v8_results, "v2": v2_results}, f, indent=2)


if __name__ == "__main__":
    main()
