#!/usr/bin/env python3
"""LLM-as-judge validation of contamination detector outputs.

This script samples items from Phase 4 audit results (balanced: 50%
high-risk and 50% clean per model), shows each item text to one or more
LLM judges WITHOUT revealing the detector's flag, and collects a
rubric-scored judgment. Inter-judge kappa + agreement with the detector
are then computed.

The output is reported in the paper as "LLM-as-judge validation" — NOT
as human evaluation. Do not mislabel.

Usage (requires ANTHROPIC_API_KEY for the Claude judge):
    python llm_judge.py --n_items 100 --seed 42 --judges claude,gpt4o
"""
import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np

PROJECT = "{REPO_ROOT}"
PHASE4 = f"{PROJECT}/results/phase4"      # switch to phase4_v2 after v2 reapply
OUT_DIR = f"{PROJECT}/results/llm_judge"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

RUBRIC = """You are reviewing whether a given item has likely been seen during the \
pretraining of a language model. Consider clues like:
- Formatting / text that matches common web or book scrapes
- Distinctive phrasings that would also appear in training data
- Items that look like standard competition problems, canonical examples, \
or well-known benchmarks
Rate the likelihood that THIS item was seen during training of the \
named model family on a 1-5 scale:
  1 = very unlikely to be in training data
  2 = unlikely
  3 = unclear
  4 = likely
  5 = very likely
Give ONE sentence of reasoning, then the rating on its own line like:
REASON: <your sentence>
RATING: <1-5>
"""


def sample_items(per_model: int = 10, seed: int = 42):
    """Sample balanced high-risk / clean items from each model's Phase 4 results."""
    rnd = random.Random(seed)
    samples = []
    for m in sorted(os.listdir(PHASE4)):
        if m.startswith("_"):
            continue
        sp = f"{PHASE4}/{m}/scores.pkl"
        if not os.path.exists(sp):
            continue
        with open(sp, "rb") as f:
            data = pickle.load(f)
        p = np.asarray(data["p_contam"])
        prov = data["provenance"]
        high = np.where(p > 0.5)[0].tolist()
        low = np.where(p < 0.1)[0].tolist()
        rnd.shuffle(high); rnd.shuffle(low)
        k = per_model // 2
        for idx in high[:k]:
            samples.append({"model": m.replace("__", "/"),
                             "benchmark": prov[idx][0],
                             "item_idx": prov[idx][1],
                             "p_contam": float(p[idx]),
                             "group": "high_risk"})
        for idx in low[:k]:
            samples.append({"model": m.replace("__", "/"),
                             "benchmark": prov[idx][0],
                             "item_idx": prov[idx][1],
                             "p_contam": float(p[idx]),
                             "group": "clean"})
    rnd.shuffle(samples)
    return samples


def load_item_text(benchmark: str, item_idx: int) -> str:
    """Reconstruct the rendered text for a given benchmark item."""
    import sys
    sys.path.insert(0, f"{PROJECT}/code")
    from detectors import render_benchmark_item
    path = f"/mnt/localssd/data/benchmarks/{benchmark}.jsonl"
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    if item_idx >= len(items):
        return ""
    return render_benchmark_item(benchmark, items[item_idx])


def parse_rating(text: str):
    rating = None
    reason = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("RATING:"):
            try:
                rating = int(line.split(":", 1)[1].strip()[0])
            except Exception:
                rating = None
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return rating, reason


def judge_claude(prompt: str, item_text: str, model_name: str):
    """Ask Claude Sonnet for a rating. Requires ANTHROPIC_API_KEY."""
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nModel family: {model_name}\nItem:\n<<<\n{item_text[:2000]}\n>>>",
        }],
    )
    return msg.content[0].text if msg.content else ""


def judge_openai(prompt: str, item_text: str, model_name: str):
    """Ask GPT-4o. Requires OPENAI_API_KEY."""
    try:
        import openai
    except ImportError:
        return None
    client = openai.OpenAI()
    r = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nModel family: {model_name}\nItem:\n<<<\n{item_text[:2000]}\n>>>",
        }],
    )
    return r.choices[0].message.content


JUDGES = {"claude": judge_claude, "openai": judge_openai}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_items", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judges", default="claude",
                     help="Comma-separated: claude, openai")
    args = ap.parse_args()

    n_models = sum(1 for m in os.listdir(PHASE4) if not m.startswith("_")
                    and os.path.exists(f"{PHASE4}/{m}/scores.pkl"))
    per_model = max(2, args.n_items // max(n_models, 1))
    samples = sample_items(per_model=per_model, seed=args.seed)[:args.n_items]
    print(f"Sampled {len(samples)} items across {n_models} models")

    judges = [j.strip() for j in args.judges.split(",") if j.strip() in JUDGES]
    print(f"Judges: {judges}")

    results = []
    for i, s in enumerate(samples):
        txt = load_item_text(s["benchmark"], s["item_idx"])
        if not txt:
            continue
        rec = dict(s, item_text=txt[:500], ratings={}, reasons={})
        for j in judges:
            try:
                resp = JUDGES[j](RUBRIC, txt, s["model"])
                rating, reason = parse_rating(resp or "")
                rec["ratings"][j] = rating
                rec["reasons"][j] = reason
            except Exception as e:
                rec["ratings"][j] = None
                rec["reasons"][j] = f"ERROR: {type(e).__name__}: {e}"
        results.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} items")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = f"{OUT_DIR}/judgments_{ts}.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[SAVED] {out}")

    # Quick aggregate stats
    from sklearn.metrics import cohen_kappa_score
    detector_labels = [1 if r["group"] == "high_risk" else 0 for r in results]
    for j in judges:
        jr = [r["ratings"].get(j) for r in results]
        valid = [(d, x) for d, x in zip(detector_labels, jr) if x is not None]
        if len(valid) < 5:
            print(f"  {j}: too few valid ratings")
            continue
        d_arr = np.array([d for d, _ in valid])
        j_arr = np.array([int(x >= 4) for _, x in valid])  # >=4 = "likely in training"
        try:
            kappa = float(cohen_kappa_score(d_arr, j_arr))
        except Exception:
            kappa = None
        agree = float((d_arr == j_arr).mean())
        print(f"  judge={j}: agreement_with_detector={agree:.3f}  kappa={kappa}")


if __name__ == "__main__":
    main()
