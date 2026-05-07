#!/usr/bin/env python3
"""LLM-as-judge contamination validation using Bedrock Claude + local model.

Samples 200 items balanced high-risk/clean across audit models under v2.
Three blind judges:
  1. Claude Haiku 4.5 (via AWS Bedrock)
  2. Claude Sonnet 4.6 (via AWS Bedrock)
  3. Gemma-2-27B (local, via vLLM)

Each judge rates 1-5 whether the item plausibly appeared in the named
model family's training data, WITHOUT seeing the detector score. We then
compute inter-judge kappa, agreement with the v2 detector, and balanced
accuracy against the detector label.

Reported as "LLM-as-judge (3 diverse models, blind)" in the paper.
NEVER labeled as human evaluation.
"""
import argparse
import json
import os
import pickle
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

PROJECT = "{REPO_ROOT}"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
OUT_DIR = f"{PROJECT}/results/llm_judge"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

RUBRIC = """You are reviewing whether a given benchmark item has likely been seen during the pre-training of a language model. Consider clues like:
- Does the exact text, problem, or canonical solution look famous/widely-circulated?
- Does the format match something that would be scraped from the web (Stack Overflow, textbooks, math contests, Codeforces)?
- Does the phrasing match standard benchmark templates that would appear verbatim in training corpora?

Rate the likelihood that this specific item was seen during training of the named model family on a 1-5 scale:
1 = very unlikely to be in training data
2 = unlikely
3 = unclear
4 = likely
5 = very likely

Output exactly two lines:
REASON: <one sentence of reasoning>
RATING: <1-5 digit>
"""

BEDROCK_MODELS = {
    "haiku45": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet45": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}
BEDROCK_REGION = "us-east-1"


def render_item(benchmark: str, idx: int) -> str:
    path = f"{BENCH_DIR}/{benchmark}.jsonl"
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    if idx >= len(items):
        return ""
    import sys
    sys.path.insert(0, f"{PROJECT}/code")
    from detectors import render_benchmark_item
    return render_benchmark_item(benchmark, items[idx])


def bedrock_judge(model_key: str, prompt: str, item_text: str, model_family: str,
                   max_tokens: int = 150):
    tok = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
    model_id = BEDROCK_MODELS[model_key]
    url = f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com/model/{model_id}/invoke"
    msg = f"{prompt}\n\nModel family: {model_family}\nItem:\n<<<\n{item_text[:2000]}\n>>>"
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": msg}],
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    if "content" in data and data["content"]:
        return data["content"][0]["text"]
    return ""


def parse_rating(text: str):
    """Robust parse: find RATING on its own line, or last digit 1-5 in text."""
    import re
    rating = None
    reason = ""
    if not text:
        return rating, reason
    # First try explicit RATING: prefix
    for line in text.splitlines():
        line = line.strip()
        U = line.upper()
        if "RATING" in U and any(c.isdigit() for c in line):
            m = re.search(r"RATING[^0-9]*([1-5])", line, re.IGNORECASE)
            if m:
                rating = int(m.group(1))
        if U.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    # Fallback: last standalone 1-5 digit in text
    if rating is None:
        digits = re.findall(r"(?<!\d)([1-5])(?!\d)", text)
        if digits:
            rating = int(digits[-1])
    return rating, reason


def sample_items(n_items: int = 200, seed: int = 42):
    rnd = random.Random(seed)
    samples = []
    models = sorted([m for m in os.listdir(PHASE4_V2) if not m.startswith("_")])
    per_model = max(4, n_items // max(len(models), 1))
    for m in models:
        sp = f"{PHASE4_V2}/{m}/scores.pkl"
        if not os.path.exists(sp):
            continue
        with open(sp, "rb") as f:
            d = pickle.load(f)
        p = np.asarray(d["p_contam"])
        prov = d["provenance"]
        high = np.where(p > 0.5)[0].tolist()
        low = np.where(p < 0.1)[0].tolist()
        rnd.shuffle(high); rnd.shuffle(low)
        k = per_model // 2
        for idx in high[:k]:
            samples.append({"model": m.replace("__", "/"),
                             "benchmark": prov[idx][0],
                             "item_idx": int(prov[idx][1]),
                             "p_contam": float(p[idx]),
                             "group": "high_risk"})
        for idx in low[:k]:
            samples.append({"model": m.replace("__", "/"),
                             "benchmark": prov[idx][0],
                             "item_idx": int(prov[idx][1]),
                             "p_contam": float(p[idx]),
                             "group": "clean"})
    rnd.shuffle(samples)
    return samples[:n_items]


def run_bedrock_judges(samples):
    """Call Claude Haiku and Sonnet on all samples; save interim results."""
    out = []
    for i, s in enumerate(samples):
        txt = render_item(s["benchmark"], s["item_idx"])
        if not txt:
            continue
        rec = dict(s, item_text=txt[:500], ratings={}, reasons={})
        for mk in ["haiku45", "sonnet45"]:
            for attempt in range(3):
                try:
                    r = bedrock_judge(mk, RUBRIC, txt, s["model"])
                    rating, reason = parse_rating(r)
                    rec["ratings"][mk] = rating
                    rec["reasons"][mk] = reason
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    rec["ratings"][mk] = None
                    rec["reasons"][mk] = f"HTTP_{e.code}"
                    break
                except Exception as e:
                    rec["ratings"][mk] = None
                    rec["reasons"][mk] = f"ERR:{type(e).__name__}"
                    break
        out.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  bedrock {i+1}/{len(samples)}")
    return out


def run_local_judge(samples, model_key="gemma27b",
                     model_path="/mnt/localssd/models/gemma-2-27b",
                     gpu: int = 0):
    """Use vLLM on a local model as 3rd judge."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from vllm import LLM, SamplingParams
    print(f"[local-judge] loading {model_path}...")
    llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
               gpu_memory_utilization=0.85, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=200)

    prompts = []
    valid_idx = []
    for i, s in enumerate(samples):
        txt = render_item(s["benchmark"], s["item_idx"])
        if not txt:
            continue
        valid_idx.append(i)
        prompts.append(f"{RUBRIC}\n\nModel family: {s['model']}\n"
                        f"Item:\n<<<\n{txt[:2000]}\n>>>")
    outs = llm.generate(prompts, sp)
    ratings = {}
    for i, o in zip(valid_idx, outs):
        if o.outputs:
            rating, reason = parse_rating(o.outputs[0].text)
            ratings[i] = (rating, reason)
    return ratings


def compute_stats(judged):
    from sklearn.metrics import cohen_kappa_score
    detector_label = np.array([1 if r["group"] == "high_risk" else 0 for r in judged])
    # Binarize ratings: >=4 means "judge thinks this is likely training data"
    for mk in ["haiku45", "sonnet45", "gemma27b"]:
        arr = np.array([r["ratings"].get(mk) for r in judged])
        valid = arr != None
        if valid.sum() < 5:
            print(f"  {mk}: too few valid ratings ({valid.sum()})")
            continue
        bin_ = np.array([int(r >= 4) if r is not None else -1 for r in arr])
        mask = bin_ >= 0
        agree = (bin_[mask] == detector_label[mask]).mean()
        try:
            kappa = cohen_kappa_score(detector_label[mask], bin_[mask])
        except Exception:
            kappa = None
        # sensitivity on high-risk items
        hr_mask = mask & (detector_label == 1)
        cl_mask = mask & (detector_label == 0)
        tpr = bin_[hr_mask].mean() if hr_mask.sum() else None
        fpr = bin_[cl_mask].mean() if cl_mask.sum() else None
        print(f"  {mk:<10s}  n_valid={int(mask.sum()):>3d}  "
              f"agree={agree:.3f}  kappa={kappa:.3f}  "
              f"TPR(high-risk flagged)={tpr:.3f}  FPR(clean flagged)={fpr:.3f}")

    # Inter-judge kappa
    pairs = [("haiku45", "sonnet45"), ("haiku45", "gemma27b"),
             ("sonnet45", "gemma27b")]
    print("\n  inter-judge kappa:")
    for a, b in pairs:
        ra = np.array([r["ratings"].get(a) for r in judged])
        rb = np.array([r["ratings"].get(b) for r in judged])
        valid = np.array([x is not None and y is not None for x, y in zip(ra, rb)])
        if valid.sum() < 5:
            continue
        ba = np.array([int(r >= 4) if r is not None else 0 for r in ra])[valid]
        bb = np.array([int(r >= 4) if r is not None else 0 for r in rb])[valid]
        try:
            k = cohen_kappa_score(ba, bb)
            agree = (ba == bb).mean()
        except Exception:
            k, agree = None, None
        print(f"    {a:<10s} <> {b:<10s}  n={int(valid.sum()):>3d}  agree={agree:.3f}  kappa={k:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_local", action="store_true")
    args = ap.parse_args()

    print(f"Sampling {args.n_items} items (balanced high-risk/clean)...")
    samples = sample_items(args.n_items, args.seed)
    print(f"Actual: {len(samples)} ({sum(s['group']=='high_risk' for s in samples)} high-risk, "
          f"{sum(s['group']=='clean' for s in samples)} clean)")

    print("\n=== Bedrock Claude judges (Haiku 4.5 + Sonnet 4.6) ===")
    judged = run_bedrock_judges(samples)

    if not args.skip_local:
        print("\n=== Local judge (Gemma-2-27B via vLLM) ===")
        local_ratings = run_local_judge(samples)
        for i, r in enumerate(judged):
            if i in local_ratings:
                rating, reason = local_ratings[i]
                r["ratings"]["gemma27b"] = rating
                r["reasons"]["gemma27b"] = reason

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"{OUT_DIR}/judgments_{ts}.jsonl"
    with open(out_path, "w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    print(f"\n[SAVED] {out_path}")

    print("\n=== STATISTICS ===")
    compute_stats(judged)


if __name__ == "__main__":
    main()
