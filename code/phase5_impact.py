#!/usr/bin/env python3
"""Phase 5 — Impact analysis.

For each (audit_model, benchmark), evaluate accuracy on three subsets:
  - full:         all items
  - clean:        items with p_contam < 0.1
  - contaminated: items with p_contam > 0.5

Report accuracy drop (full vs clean) as evidence of contamination-driven inflation.

Uses vLLM for fast batched generation/classification per benchmark.

Usage:
    python phase5_impact.py --model_id EleutherAI/pythia-1b \\
                             --model_path /mnt/localssd/models/EleutherAI__pythia-1b \\
                             --benchmarks gsm8k_test,mmlu_test,humaneval
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = "{REPO_ROOT}"
LOCALSSD = "/mnt/localssd"
BENCH_DIR = f"{LOCALSSD}/data/benchmarks"
PHASE4_DIR = f"{PROJECT}/results/phase4"
RESULTS = f"{PROJECT}/results/phase5"
Path(RESULTS).mkdir(parents=True, exist_ok=True)


def load_bench(name):
    path = f"{BENCH_DIR}/{name}.jsonl"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---- Benchmark-specific evaluators ----
# Each returns a list of (pred, correct) pairs given (items, model_gen_fn).

def eval_mc_letter(items, gen_fn, prompt_tpl, correct_key):
    """For MC benchmarks: prompt, check if generated text starts with correct letter."""
    prompts = [prompt_tpl(ex) for ex in items]
    preds = gen_fn(prompts, max_tokens=6, stop=["\n"])
    results = []
    for p, ex in zip(preds, items):
        correct = str(ex[correct_key]).strip().upper()[:1]
        pred_letter = p.strip().upper()[:1] if p else ""
        results.append((pred_letter, correct, int(pred_letter == correct)))
    return results


def eval_gsm8k(items, gen_fn):
    prompts = [f"Question: {ex['question']}\nAnswer:" for ex in items]
    preds = gen_fn(prompts, max_tokens=256, stop=["\nQuestion:"])
    results = []
    for p, ex in zip(preds, items):
        # Extract final number from gold answer
        gold = ex['answer']
        gold_num = re.findall(r"-?\d[\d,]*\.?\d*", gold.split("####")[-1])[0].replace(",", "") if "####" in gold else ""
        pred_num = ""
        nums = re.findall(r"-?\d[\d,]*\.?\d*", p or "")
        if nums:
            pred_num = nums[-1].replace(",", "")
        correct = int(pred_num == gold_num) if gold_num else 0
        results.append((pred_num, gold_num, correct))
    return results


def eval_mmlu(items, gen_fn):
    letters = ["A", "B", "C", "D"]
    def _tpl(ex):
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex.get("choices", [])))
        return f"Subject: {ex.get('subject','')}\nQuestion: {ex['question']}\n{body}\nAnswer:"
    prompts = [_tpl(ex) for ex in items]
    preds = gen_fn(prompts, max_tokens=3, stop=["\n"])
    results = []
    for p, ex in zip(preds, items):
        correct = letters[ex["answer"]] if isinstance(ex["answer"], int) else ex["answer"]
        pred_letter = (p.strip().upper()[:1] if p else "")
        results.append((pred_letter, correct, int(pred_letter == correct)))
    return results


def eval_hellaswag(items, gen_fn):
    """HellaSwag is typically scored via log-likelihood ranking; approximate with
    greedy completion that should start with the correct ending's first word."""
    letters = ["A", "B", "C", "D"]
    def _tpl(ex):
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex.get("endings", [])))
        return f"Context: {ex['ctx']}\nChoose the best continuation:\n{body}\nAnswer:"
    prompts = [_tpl(ex) for ex in items]
    preds = gen_fn(prompts, max_tokens=3, stop=["\n"])
    results = []
    for p, ex in zip(preds, items):
        correct = letters[int(ex["label"])]
        pred_letter = (p.strip().upper()[:1] if p else "")
        results.append((pred_letter, correct, int(pred_letter == correct)))
    return results


def eval_arc(items, gen_fn):
    def _tpl(ex):
        choices = ex["choices"]
        labels, texts = choices["label"], choices["text"]
        body = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        return f"Question: {ex['question']}\n{body}\nAnswer:"
    prompts = [_tpl(ex) for ex in items]
    preds = gen_fn(prompts, max_tokens=3, stop=["\n"])
    results = []
    for p, ex in zip(preds, items):
        correct = str(ex["answerKey"]).strip().upper()[:1]
        pred_letter = (p.strip().upper()[:1] if p else "")
        results.append((pred_letter, correct, int(pred_letter == correct)))
    return results


def eval_humaneval(items, gen_fn):
    """Exact-match on the canonical_solution prefix (approximate; full pass@1
    would need sandboxed execution)."""
    prompts = [ex['prompt'] for ex in items]
    preds = gen_fn(prompts, max_tokens=512, stop=["\ndef ", "\nclass ", "\nif __name__"])
    results = []
    for p, ex in zip(preds, items):
        gold = ex.get('canonical_solution', '').strip()
        # Prefix match on first 30 chars (rough proxy)
        pred_head = (p or "").strip()[:30].lower()
        gold_head = gold[:30].lower()
        correct = int(pred_head == gold_head and len(pred_head) > 5)
        results.append((pred_head, gold_head, correct))
    return results


EVALUATORS = {
    "gsm8k_test": eval_gsm8k,
    "mmlu_test": eval_mmlu,
    "hellaswag_val": eval_hellaswag,
    "arc_challenge": eval_arc,
    "humaneval": eval_humaneval,
}


def run_impact(model_id: str, model_path: str, benchmarks: list):
    short = model_id.replace("/", "__")
    # Load Phase 4 contamination scores for this model
    phase4_path = f"{PHASE4_DIR}/{short}/scores.pkl"
    if not os.path.exists(phase4_path):
        raise SystemExit(f"Phase 4 scores missing: {phase4_path}")
    with open(phase4_path, "rb") as f:
        phase4 = pickle.load(f)

    # Build per-benchmark contamination masks
    prov = phase4["provenance"]
    p_contam = phase4["p_contam"]

    out_dir = f"{RESULTS}/{short}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    print(f"[{model_id}] loading vLLM...")
    llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.9, max_model_len=2048)

    def gen_fn(prompts, max_tokens=16, stop=None):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop or [])
        outs = llm.generate(prompts, sp)
        return [o.outputs[0].text if o.outputs else "" for o in outs]

    summary = {}
    for bench in benchmarks:
        if bench not in EVALUATORS:
            print(f"[{model_id}] SKIP {bench} (no evaluator)")
            continue
        items = load_bench(bench)
        if not items:
            continue
        bench_mask = np.array([p[0] == bench for p in prov])
        if not bench_mask.any():
            continue
        bench_p = p_contam[bench_mask]

        # Align items with prov order for this benchmark
        bench_indices = [p[1] for p in prov if p[0] == bench]
        bench_items = [items[i] for i in bench_indices if i < len(items)]

        if len(bench_items) != len(bench_p):
            print(f"[{model_id}] WARN bench {bench}: {len(bench_items)} items vs {len(bench_p)} scores")
            n = min(len(bench_items), len(bench_p))
            bench_items = bench_items[:n]; bench_p = bench_p[:n]

        # Define subsets
        full_idx = np.arange(len(bench_items))
        clean_idx = np.where(bench_p < 0.1)[0]
        contam_idx = np.where(bench_p > 0.5)[0]

        subsets = {
            "full": [bench_items[i] for i in full_idx],
            "clean": [bench_items[i] for i in clean_idx],
            "contaminated": [bench_items[i] for i in contam_idx],
        }

        ev = EVALUATORS[bench]
        subset_acc = {}
        for name, subset_items in subsets.items():
            if not subset_items:
                subset_acc[name] = {"n": 0, "acc": None}
                continue
            t0 = time.time()
            results = ev(subset_items, gen_fn)
            acc = np.mean([r[-1] for r in results])
            subset_acc[name] = {"n": len(subset_items), "acc": float(acc),
                                 "duration_sec": time.time() - t0}
            print(f"[{model_id}] {bench}/{name}: n={len(subset_items)}, acc={acc:.3f}")

        summary[bench] = subset_acc

    with open(f"{out_dir}/impact.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {out_dir}/impact.json")

    # Experiment log
    entry = {
        "id": f"impact_{short}",
        "paper": "paper_5_contamination",
        "name": f"impact_{short}",
        "status": "success",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": out_dir,
        "summary": summary,
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--benchmarks", default="gsm8k_test,mmlu_test,hellaswag_val,arc_challenge,humaneval")
    args = ap.parse_args()
    run_impact(args.model_id, args.model_path, args.benchmarks.split(","))


if __name__ == "__main__":
    main()
