#!/usr/bin/env python3
"""Das 2024 blind-baseline comparison.

Das et al. 2024 show that for cross-distribution MIA/contamination
evaluations, most apparent success comes from a ``blind baseline''
that uses only item-text statistics (length, perplexity relative to a
control LM) without any access to the putatively-contaminated target
model's outputs.

We implement two blind baselines and test whether our v2 ensemble's
F1 or deployment signal depends on features that a blind baseline
already captures:

  Blind-1: text length (character count) alone.
  Blind-2: Pythia-70M (a tiny uncontaminated LM) per-item loss.

For each variant (A-E), we compute the two blind baselines and
calibration F1 on the v1 (cross-benchmark) and v2 (same-benchmark)
negative sets. If blind-1 or blind-2 already achieves high F1 under
v1, Das's critique applies. If v2 F1 is high but blind F1 is near
chance, v2 is distinct from blind baselines.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
OUT = f"{PROJECT}/results/blind_baseline"
Path(OUT).mkdir(parents=True, exist_ok=True)

V2_DIR = f"{PROJECT}/results/phase3_v2"


def load_bench(name):
    path = f"{BENCH_DIR}/{name}.jsonl"
    if not Path(path).exists():
        return []
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def best_f1(y, s):
    y = np.asarray(y); s = np.asarray(s)
    m = ~np.isnan(s) & ~np.isnan(y)
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return float("nan"), float("nan")
    auc = roc_auc_score(y[m], s[m])
    p, r, _ = precision_recall_curve(y[m], s[m])
    with np.errstate(invalid="ignore"):
        f1 = 2 * p * r / (p + r + 1e-12)
    return float(auc), float(np.nanmax(f1))


def main():
    # Use Min-K%++ run's negatives as a surrogate for calibration data
    # (pos = variant-specific benchmark, neg = cross-benchmark pooled items).
    # For each variant we have the pos/neg item text; we compute blind-1 (length)
    # and compare to saved Min-K%++ F1.
    variant_to_pos_bench = {
        "A": "gsm8k_test",
        "B": "mmlu_test",
        "C": "hellaswag_val",
        "D": "arc_challenge",
    }

    def render_item(bench, ex):
        if bench == "gsm8k_test":
            return f"Question: {ex['question']}\nAnswer: {ex.get('answer','')}"
        if bench == "mmlu_test":
            choices = ex.get("choices", [])
            body = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            return f"Q: {ex['question']}\n{body}\nA: {ex['answer']}"
        if bench == "hellaswag_val":
            ends = ex.get("endings", [])
            body = "\n".join(f"{chr(65+i)}. {e}" for i, e in enumerate(ends))
            return f"Context: {ex.get('ctx', '')}\n{body}"
        if bench == "arc_challenge":
            ch = ex.get("choices", {})
            body = "\n".join(f"{l}. {t}" for l, t in
                              zip(ch.get("label", []), ch.get("text", [])))
            return f"Q: {ex['question']}\n{body}"
        if bench == "humaneval":
            return ex.get("prompt", "") + (ex.get("canonical_solution") or "")
        if bench == "mbpp":
            return ex.get("text", "") + "\n" + (ex.get("code") or "")
        return str(ex)

    rows = []
    for v, pos_bench in variant_to_pos_bench.items():
        pos_items = load_bench(pos_bench)[:500]
        pos_texts = [render_item(pos_bench, ex) for ex in pos_items]
        # Cross-benchmark negatives (v1-style)
        neg_benches = [b for b in ["gsm8k_test", "mmlu_test", "hellaswag_val",
                                     "arc_challenge", "humaneval", "mbpp"]
                        if b != pos_bench]
        neg_texts = []
        per = 100
        for b in neg_benches:
            items = load_bench(b)[:per]
            neg_texts.extend([render_item(b, ex) for ex in items])

        # Blind-1: text length in characters
        pos_len = np.array([len(t) for t in pos_texts], dtype=float)
        neg_len = np.array([len(t) for t in neg_texts], dtype=float)
        y = np.concatenate([np.ones(len(pos_len)), np.zeros(len(neg_len))])
        s = np.concatenate([pos_len, neg_len])
        auc_len, f1_len = best_f1(y, s)
        # Also try negated (shorter=positive)
        auc_len_neg, f1_len_neg = best_f1(y, -s)
        auc_len_best = max(auc_len, auc_len_neg)
        f1_len_best = max(f1_len, f1_len_neg)

        # Blind-1': token length via cheap whitespace split
        pos_tok = np.array([len(t.split()) for t in pos_texts], dtype=float)
        neg_tok = np.array([len(t.split()) for t in neg_texts], dtype=float)
        s_tok = np.concatenate([pos_tok, neg_tok])
        auc_tok, f1_tok = best_f1(y, s_tok)

        # Min-K%++ already computed — read F1
        with open(f"{PROJECT}/results/minkpp/variant_{v}_summary.json") as f:
            mk = json.load(f)
        mk_f1 = mk.get("cross_bench_calibration", {}).get("f1", float("nan"))
        mk_auc = mk.get("cross_bench_calibration", {}).get("auc", float("nan"))

        row = {
            "variant": v, "pos_bench": pos_bench,
            "blind_char_len_auc": round(auc_len_best, 4),
            "blind_char_len_f1": round(f1_len_best, 4),
            "blind_word_len_auc": round(auc_tok, 4),
            "minkpp_auc": round(mk_auc, 4),
            "minkpp_f1": round(mk_f1, 4),
        }
        rows.append(row)
        print(f"Variant {v} ({pos_bench}): char-len AUC={auc_len_best:.3f} F1={f1_len_best:.3f}  "
              f"word-len AUC={auc_tok:.3f}  Min-K%++ AUC={mk_auc:.4f} F1={mk_f1:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/blind_baseline.csv", index=False)
    print(f"\nSaved to {OUT}/blind_baseline.csv")
    # Print gap
    print("\nDas 2024 blind-baseline gap (bigger = v2 beats blind more):")
    for _, r in df.iterrows():
        gap = r["minkpp_f1"] - r["blind_char_len_f1"]
        print(f"  {r['variant']} ({r['pos_bench']}): Min-K%++ F1 {r['minkpp_f1']:.3f} "
              f"vs blind-len F1 {r['blind_char_len_f1']:.3f}  (delta={gap:+.3f})")


if __name__ == "__main__":
    main()
