#!/usr/bin/env python3
"""Oren et al. 2024 proving-test (option-shuffle variant) on Claude.

The original Oren et al. proving-test uses logprobs to compare canonical
ordering against permuted orderings. Closed-API models do not expose
token-level logprobs, so we use a generation-based proxy:

For each multiple-choice item we record (a) canonical accuracy with
the original option ordering and (b) shuffled accuracy where the same
options are presented in a deterministic permuted order. A model that
has memorized the canonical ordering of a benchmark item will tend to
answer the canonical version correctly while being more confused on
the shuffled version. A model that reasons about the item should be
roughly insensitive to the order. The signature of contamination is a
positive (canonical - shuffled) gap.

We run this on Claude Haiku 4.5 and Sonnet 4.5 across MMLU and ARC
on 200 items each.

Output: one JSON file per (model, benchmark) under results/oren/ with
per-item canonical / shuffled / labels and an aggregate t-test on the
gap.
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
OUT = f"{PROJECT}/results/oren_claude"
Path(OUT).mkdir(parents=True, exist_ok=True)

BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/invoke"
MODELS = {
    "haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def call_bedrock(model_id, user_content, max_tokens=20, max_retries=5):
    url = BEDROCK_URL.format(model_id=model_id)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_content}],
    })
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=body.encode(), method="POST",
                headers={
                    "Authorization": f"Bearer {os.environ['AWS_BEARER_TOKEN_BEDROCK']}",
                    "Content-Type": "application/json",
                })
            r = urllib.request.urlopen(req, timeout=60)
            resp = json.loads(r.read())
            text = resp.get("content", [{}])[0].get("text", "")
            return text
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                time.sleep(2 ** attempt + np.random.rand())
                continue
            return f"ERR_HTTP{e.code}"
        except Exception:
            time.sleep(2 ** attempt + np.random.rand())
    return "ERR_MAX_RETRIES"


def load_bench(bench, n):
    items = []
    with open(f"{BENCH_DIR}/{bench}.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def make_canonical_prompt_mmlu(item):
    body = "\n".join(
        f"{LETTERS[i]}. {c}"
        for i, c in enumerate(item.get("choices", [])[:26]))
    return (f"Answer with only the letter (A-D).\n\n"
            f"Subject: {item.get('subject','')}\n"
            f"{item['question']}\n{body}")


def make_shuffled_prompt_mmlu(item, seed):
    """Permute the option labels deterministically by item index seed.
    Track which permuted label corresponds to the gold answer.
    """
    choices = list(item.get("choices", [])[:26])
    n = len(choices)
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    # perm[i] tells us which original choice goes into permuted slot i
    body = "\n".join(f"{LETTERS[i]}. {choices[perm[i]]}" for i in range(n))
    gold_orig = item["answer"] if isinstance(item["answer"], int) else 0
    gold_new = perm.index(gold_orig)
    return (f"Answer with only the letter (A-D).\n\n"
            f"Subject: {item.get('subject','')}\n"
            f"{item['question']}\n{body}"), gold_new


def make_canonical_prompt_arc(item):
    ch = item["choices"]
    body = "\n".join(f"{l}. {t}" for l, t in zip(ch["label"], ch["text"]))
    return f"Answer with only the letter.\n\n{item['question']}\n{body}"


def make_shuffled_prompt_arc(item, seed):
    ch = item["choices"]
    labels = list(ch["label"])
    texts = list(ch["text"])
    n = len(labels)
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    body = "\n".join(
        f"{labels[i]}. {texts[perm[i]]}" for i in range(n))
    gold_label = str(item["answerKey"]).strip().upper()[:1]
    if gold_label in labels:
        gold_orig = labels.index(gold_label)
        gold_new = perm.index(gold_orig)
        gold_new_label = labels[gold_new]
    else:
        gold_new_label = gold_label
    return (f"Answer with only the letter.\n\n{item['question']}\n{body}"),\
            gold_new_label


def score_letter(response, gold):
    if response.startswith("ERR_"):
        return None
    r = response.strip().upper()
    pred = r[:1] if r else ""
    if isinstance(gold, int):
        return int(pred == LETTERS[gold])
    return int(pred == str(gold).upper()[:1])


def run_pair(model_short, bench, n, seed_base=12345):
    items = load_bench(bench, n)
    model_id = MODELS[model_short]
    rows = []
    t0 = time.time()
    for i, item in enumerate(items):
        if bench == "mmlu_test":
            p_canon = make_canonical_prompt_mmlu(item)
            p_shuf, gold_shuf = make_shuffled_prompt_mmlu(item, seed_base + i)
            gold_canon = item["answer"]
        elif bench == "arc_challenge":
            p_canon = make_canonical_prompt_arc(item)
            p_shuf, gold_shuf = make_shuffled_prompt_arc(item, seed_base + i)
            gold_canon = str(item["answerKey"]).strip().upper()[:1]
        else:
            raise ValueError(bench)
        r_canon = call_bedrock(model_id, p_canon, max_tokens=10)
        r_shuf = call_bedrock(model_id, p_shuf, max_tokens=10)
        c_canon = score_letter(r_canon, gold_canon)
        c_shuf = score_letter(r_shuf, gold_shuf)
        rows.append({
            "i": i,
            "canon_correct": c_canon,
            "shuf_correct": c_shuf,
            "canon_resp": r_canon[:30],
            "shuf_resp": r_shuf[:30],
            "gold_canon": gold_canon if isinstance(gold_canon, int)
                              else str(gold_canon),
            "gold_shuf": gold_shuf if isinstance(gold_shuf, int)
                              else str(gold_shuf),
        })
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {model_short} {bench}: {i+1}/{n} "
                   f"({elapsed:.0f}s, {(i+1)/elapsed:.2f}/s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--benches", nargs="+",
                     default=["mmlu_test", "arc_challenge"])
    args = ap.parse_args()

    if "AWS_BEARER_TOKEN_BEDROCK" not in os.environ:
        sys.exit("AWS_BEARER_TOKEN_BEDROCK env var required")

    print(f"Running Oren proving-test (option-shuffle variant): "
          f"models={args.models}, benches={args.benches}, n={args.n}",
          flush=True)
    summary = {}
    for m in args.models:
        for b in args.benches:
            tag = f"{m}__{b}"
            out = f"{OUT}/{tag}.json"
            if os.path.exists(out):
                rows = json.load(open(out))
                print(f"Loaded cached {tag}: n={len(rows)}", flush=True)
            else:
                rows = run_pair(m, b, args.n)
                json.dump(rows, open(out, "w"))
                print(f"Wrote {out}: n={len(rows)}", flush=True)
            valid = [r for r in rows
                       if r["canon_correct"] is not None
                       and r["shuf_correct"] is not None]
            if not valid:
                continue
            canon = np.array([r["canon_correct"] for r in valid])
            shuf = np.array([r["shuf_correct"] for r in valid])
            gap = canon - shuf
            mean_gap = float(gap.mean())
            t_stat, p_val = stats.ttest_1samp(gap, 0)
            summary[tag] = {
                "n": len(valid),
                "canon_acc": float(canon.mean()),
                "shuf_acc": float(shuf.mean()),
                "mean_gap": mean_gap,
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            }
            print(f"  {tag}: canon={canon.mean():.3f}, "
                   f"shuf={shuf.mean():.3f}, "
                   f"gap={mean_gap:+.3f}, p={p_val:.4f}", flush=True)

    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print(f"\nSummary saved to {OUT}/summary.json", flush=True)


if __name__ == "__main__":
    main()
