#!/usr/bin/env python3
"""Evaluate pass@1 on the MATH benchmark with numeric-answer extraction.

Gold answers are in LaTeX \\boxed{...}. We extract the boxed content and
compare numerically after normalization (fractions, integers, simple
symbolic forms). This is an approximation of the standard MATH evaluator
(e.g., Hendrycks' original grader); it errs conservative (some correct
symbolic equivalents may be missed) but is consistent across models so
the clean-vs-contaminated gap comparison remains valid.
"""
import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
RESULTS = f"{PROJECT}/results/math_eval"
Path(RESULTS).mkdir(parents=True, exist_ok=True)


def extract_boxed(text: str):
    """Find last \\boxed{...} content, or last number if no box."""
    if not text:
        return None
    # Find \boxed{...}, handling nested braces
    idx = text.rfind("\\boxed{")
    if idx >= 0:
        depth = 0
        start = idx + len("\\boxed{")
        end = start
        while end < len(text):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                if depth == 0:
                    break
                depth -= 1
            end += 1
        if end < len(text):
            return text[start:end].strip()
    # Fallback: last numeric in text
    m = re.findall(r"-?\d[\d,/.]*", text)
    if m:
        return m[-1].replace(",", "").strip()
    return None


def normalize_answer(a: str):
    if a is None:
        return None
    a = a.strip().replace("$", "").replace(" ", "")
    # Remove \\frac{a}{b} → a/b
    a = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", a)
    a = a.replace("\\", "")
    # Try parse as number
    try:
        if "/" in a and a.count("/") == 1:
            p, q = a.split("/")
            return float(p) / float(q)
        return float(a)
    except Exception:
        return a.lower()


def match(pred, gold):
    np_ = normalize_answer(pred); ng = normalize_answer(gold)
    if np_ is None or ng is None:
        return 0
    if isinstance(np_, float) and isinstance(ng, float):
        return int(abs(np_ - ng) < 1e-4)
    return int(str(np_) == str(ng))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--benchmark", default="math",
                     help="math or math500")
    args = ap.parse_args()

    items = []
    with open(f"{BENCH_DIR}/{args.benchmark}.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"[{args.model_id}] {args.benchmark}: {len(items)} items")

    from vllm import LLM, SamplingParams
    import os as _os
    tp = int(_os.environ.get("TP", "1"))
    eager = tp > 1
    print(f"[{args.model_id}] loading vLLM (tp={tp} eager={eager})...")
    llm = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True,
               gpu_memory_utilization=0.9, max_model_len=2048,
               tensor_parallel_size=tp, enforce_eager=eager)
    sp = SamplingParams(temperature=0.0, max_tokens=512,
                         stop=["\nProblem:"])

    prompts = []
    for ex in items:
        prob = ex.get("problem") or ex.get("question") or ""
        prompts.append(
            f"Problem: {prob}\nSolution: Let's solve step by step. "
            f"Give the final answer in \\boxed{{}}.\n")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[{args.model_id}] gen in {time.time()-t0:.0f}s")

    results = []
    for ex, o in zip(items, outputs):
        comp = o.outputs[0].text if o.outputs else ""
        pred = extract_boxed(comp)
        gold_sol = ex.get("solution") or ""
        gold = extract_boxed(gold_sol)
        # MATH-500 has separate 'answer' field too
        if gold is None and "answer" in ex:
            gold = str(ex["answer"])
        correct = match(pred, gold)
        results.append({"pred": pred, "gold": gold, "correct": correct})
    acc = float(np.mean([r["correct"] for r in results]))
    short = args.model_id.replace("/", "__")
    out_dir = f"{RESULTS}/{short}_{args.benchmark}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{out_dir}/aggregate.json", "w") as f:
        json.dump({"accuracy": acc, "n": len(results),
                    "benchmark": args.benchmark}, f, indent=2)
    print(f"[{args.model_id}] {args.benchmark} acc = {acc:.3f}")

    entry = {
        "id": f"math_eval_{short}_{args.benchmark}",
        "paper": "paper_5_contamination",
        "status": "success",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": out_dir,
        "accuracy": acc,
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


if __name__ == "__main__":
    main()
