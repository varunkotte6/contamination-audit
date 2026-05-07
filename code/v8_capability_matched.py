#!/usr/bin/env python3
"""v8: capability-matched pairwise reference detector for MATH-500.

For each audited model m, select K reference models r_1, ..., r_K whose
MATH-500 accuracy is closest to m's but not m itself. Compute
    delta(m, i) = answer_loss(m, i) - mean_k(answer_loss(r_k, i))
The resulting delta is anomalously negative on items where m's
answer-span loss is low RELATIVE TO CAPABILITY PEERS, which is the
signature of item-specific memorization beyond general capability.

Capability matching (based on phase5_full raw MATH-500 accuracy) removes
the confound where a highly-capable model m has systematically lower
answer_loss than arbitrary peers.

Stratified gap under v8:
- CONTAM: delta <= tau_low (m is anomalously confident vs capable peers)
- CLEAN:  delta >= tau_high (m is no more confident than peers)
- Gap = acc(clean) - acc(contam). A working detector: gap < 0.
"""
import json
import os
import pickle

import numpy as np

PROJECT = "{REPO_ROOT}"
ANS_DIR = f"{PROJECT}/results/v6_answer_loss"
MATH_EVAL = f"{PROJECT}/results/math_eval"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"


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


def load_all():
    slugs = sorted([f[:-4] for f in os.listdir(ANS_DIR) if f.endswith(".npz")])
    all_al = {}
    all_correct = {}
    for s in slugs:
        al = np.load(f"{ANS_DIR}/{s}.npz")["answer_loss"]
        all_al[s] = al
        c = load_math500_correct(s)
        if c is not None:
            all_correct[s] = c
    return slugs, all_al, all_correct


def model_accuracy(all_correct, s):
    c = all_correct.get(s)
    if c is None:
        return None
    return float(c.mean())


def main():
    slugs, all_al, all_correct = load_all()
    accs = {s: model_accuracy(all_correct, s) for s in slugs}
    print(f"Loaded {len(slugs)} models with answer_loss")
    print("MATH-500 accuracy per model:")
    for s, a in sorted(accs.items(), key=lambda kv: (kv[1] is None, kv[1])):
        print(f"  {s:<45s} {a if a is None else f'{a:.3f}'}")

    for K in [2, 3, 5]:
        for tail_q in [0.2, 0.25]:
            print(f"\n=== K={K} refs, tail={tail_q} ===")
            print(f"{'model':<45s} {'acc':>5s} "
                  f"{'n_cl':>4s} {'n_ct':>4s} "
                  f"{'cl_acc':>6s} {'ct_acc':>6s} {'v8_gap':>7s} {'v2_gap':>7s}")
            v8_gaps = []; v2_gaps = []
            for m in slugs:
                if m not in all_correct:
                    continue
                m_al = all_al[m]
                m_correct = all_correct[m]
                m_acc = accs[m]
                # Rank other models by |acc - m_acc|
                others = [(s, abs(accs[s] - m_acc)) for s in slugs
                           if s != m and s in all_correct and accs[s] is not None]
                others.sort(key=lambda kv: kv[1])
                ref_slugs = [s for s, _ in others[:K]]
                if len(ref_slugs) < K:
                    continue
                ref_al = np.stack([all_al[s] for s in ref_slugs], axis=0)
                n = min(len(m_al), ref_al.shape[1], len(m_correct))
                m_al_n = m_al[:n]; ref_al_n = ref_al[:, :n]; c_n = m_correct[:n]
                valid = ~np.isnan(m_al_n) & ~np.isnan(ref_al_n).any(axis=0)
                if valid.sum() < 30:
                    continue
                delta = m_al_n[valid] - ref_al_n[:, valid].mean(axis=0)
                correct_v = c_n[valid]
                tau_lo = np.quantile(delta, tail_q)
                tau_hi = np.quantile(delta, 1 - tail_q)
                contam = delta <= tau_lo
                clean = delta >= tau_hi
                if contam.sum() < 10 or clean.sum() < 10:
                    continue
                cl_acc = float(correct_v[clean].mean())
                ct_acc = float(correct_v[contam].mean())
                gap = cl_acc - ct_acc
                v8_gaps.append(gap)
                # v2 comparison
                p_v2 = load_v2_pcontam(m)
                v2s = "   nan"
                if p_v2 is not None:
                    p_v2 = p_v2[:n][valid]
                    v2_clean = p_v2 < 0.1; v2_contam = p_v2 > 0.5
                    if v2_clean.sum() >= 10 and v2_contam.sum() >= 10:
                        v2_gap = float(correct_v[v2_clean].mean()
                                        - correct_v[v2_contam].mean())
                        v2_gaps.append(v2_gap)
                        v2s = f"{v2_gap:+.3f}"
                sign = "CORR" if gap < 0 else "wrng"
                print(f"{m:<45s} {m_acc:>.3f} "
                       f"{int(clean.sum()):>4d} {int(contam.sum()):>4d} "
                       f"{cl_acc:>+.3f} {ct_acc:>+.3f} "
                       f"{gap:>+.3f}[{sign}] {v2s:>7s}")
            if v8_gaps:
                n_neg = sum(1 for g in v8_gaps if g < 0)
                print(f"v8 mean_gap={np.mean(v8_gaps):+.4f} median={np.median(v8_gaps):+.4f} "
                      f"sign_neg={n_neg}/{len(v8_gaps)}")
            if v2_gaps:
                n_neg = sum(1 for g in v2_gaps if g < 0)
                print(f"v2 mean_gap={np.mean(v2_gaps):+.4f} median={np.median(v2_gaps):+.4f} "
                      f"sign_neg={n_neg}/{len(v2_gaps)}")


if __name__ == "__main__":
    main()
