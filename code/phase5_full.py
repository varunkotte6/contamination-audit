#!/usr/bin/env python3
"""Phase 5 full — per-item accuracy evaluation on every benchmark.

Unlike the earlier phase5_impact.py (which evaluated per-subset and only
stored subset-level accuracy), this saves per-item correctness so subsets
can be re-sliced post-hoc against any ensemble version (v1, v2, lighter
regimes, etc.) without re-running generation.

Benchmarks: GSM8K, MMLU, HellaSwag, ARC-Challenge, HumanEval. (MATH and
LiveCodeBench are added in phase5_generative.py because they need
sandboxed or numeric-extraction specialized evaluators.)

Usage:
    python phase5_full.py --model_id EleutherAI/pythia-1b \\
                          --model_path /mnt/localssd/models/EleutherAI__pythia-1b
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
BENCH_DIR = "/mnt/localssd/data/benchmarks"
RESULTS = f"{PROJECT}/results/phase5_full"
Path(RESULTS).mkdir(parents=True, exist_ok=True)


def load_bench(name):
    path = f"{BENCH_DIR}/{name}.jsonl"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def eval_gsm8k(items, gen_fn):
    prompts = [f"Question: {ex['question']}\nAnswer:" for ex in items]
    preds = gen_fn(prompts, max_tokens=256, stop=["\nQuestion:"])
    out = []
    for p, ex in zip(preds, items):
        gold = ex["answer"]
        gold_num = ""
        if "####" in gold:
            m = re.findall(r"-?\d[\d,]*\.?\d*", gold.split("####")[-1])
            if m:
                gold_num = m[0].replace(",", "")
        pred_num = ""
        m = re.findall(r"-?\d[\d,]*\.?\d*", p or "")
        if m:
            pred_num = m[-1].replace(",", "")
        out.append({"pred": pred_num, "gold": gold_num,
                    "correct": int(pred_num == gold_num and gold_num != "")})
    return out


def eval_mmlu(items, gen_fn):
    letters = ["A", "B", "C", "D"]
    def _tpl(ex):
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex.get("choices", [])))
        return f"Subject: {ex.get('subject','')}\nQuestion: {ex['question']}\n{body}\nAnswer:"
    prompts = [_tpl(ex) for ex in items]
    preds = gen_fn(prompts, max_tokens=3, stop=["\n"])
    out = []
    for p, ex in zip(preds, items):
        correct = letters[ex["answer"]] if isinstance(ex["answer"], int) else ex["answer"]
        pred_letter = (p.strip().upper()[:1] if p else "")
        out.append({"pred": pred_letter, "gold": str(correct)[:1],
                    "correct": int(pred_letter == str(correct)[:1])})
    return out


def eval_hellaswag(items, gen_fn):
    letters = ["A", "B", "C", "D"]
    def _tpl(ex):
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex.get("endings", [])))
        return f"Context: {ex['ctx']}\nChoose the best continuation:\n{body}\nAnswer:"
    prompts = [_tpl(ex) for ex in items]
    preds = gen_fn(prompts, max_tokens=3, stop=["\n"])
    out = []
    for p, ex in zip(preds, items):
        correct = letters[int(ex["label"])]
        pred_letter = (p.strip().upper()[:1] if p else "")
        out.append({"pred": pred_letter, "gold": correct,
                    "correct": int(pred_letter == correct)})
    return out


def eval_arc(items, gen_fn):
    def _tpl(ex):
        ch = ex["choices"]
        body = "\n".join(f"{l}. {t}" for l, t in zip(ch["label"], ch["text"]))
        return f"Question: {ex['question']}\n{body}\nAnswer:"
    prompts = [_tpl(ex) for ex in items]
    preds = gen_fn(prompts, max_tokens=3, stop=["\n"])
    out = []
    for p, ex in zip(preds, items):
        correct = str(ex["answerKey"]).strip().upper()[:1]
        pred_letter = (p.strip().upper()[:1] if p else "")
        out.append({"pred": pred_letter, "gold": correct,
                    "correct": int(pred_letter == correct)})
    return out


def eval_humaneval(items, gen_fn):
    """Approximate pass@1 via prefix match (see paper limitations section)."""
    prompts = [ex["prompt"] for ex in items]
    preds = gen_fn(prompts, max_tokens=512,
                    stop=["\ndef ", "\nclass ", "\nif __name__"])
    out = []
    for p, ex in zip(preds, items):
        gold = (ex.get("canonical_solution") or "").strip()
        pred_head = (p or "").strip()[:30].lower()
        gold_head = gold[:30].lower()
        correct = int(pred_head == gold_head and len(pred_head) > 5)
        out.append({"pred_prefix": pred_head, "gold_prefix": gold_head,
                    "correct": correct})
    return out


def eval_mbpp(items, gen_fn):
    """MBPP prefix-match on 'code' field (approximation)."""
    prompts = [f"{ex.get('text', ex.get('prompt',''))}\n" for ex in items]
    preds = gen_fn(prompts, max_tokens=512,
                    stop=["\ndef ", "\nclass ", "\nif __name__"])
    out = []
    for p, ex in zip(preds, items):
        gold = (ex.get('code') or '').strip()
        pred_head = (p or '').strip()[:30].lower()
        gold_head = gold[:30].lower()
        correct = int(pred_head == gold_head and len(pred_head) > 5)
        out.append({"pred_prefix": pred_head, "gold_prefix": gold_head, "correct": correct})
    return out


def eval_truthfulqa(items, gen_fn):
    """TruthfulQA multiple-choice: match mc1 correct answer by first letter of choice."""
    letters = ['A','B','C','D','E','F','G','H']
    def _tpl(ex):
        mc1 = ex.get('mc1_targets', {})
        choices = mc1.get('choices', [])
        if not choices:
            return None
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:len(letters)]))
        return f"Question: {ex.get('question','')}\n{body}\nAnswer:"
    prompts_items = [(_tpl(ex), ex) for ex in items]
    valid = [(p, ex) for p, ex in prompts_items if p is not None]
    if not valid:
        return [{"correct": 0} for _ in items]
    preds = gen_fn([p for p, _ in valid], max_tokens=3, stop=["\n"])
    out_valid = []
    for p, (_, ex) in zip(preds, valid):
        mc1 = ex.get('mc1_targets', {})
        labels = mc1.get('labels', [])
        if labels and 1 in labels:
            correct_idx = labels.index(1)
            correct_letter = letters[correct_idx] if correct_idx < len(letters) else "?"
        else:
            correct_letter = "?"
        pred_letter = (p.strip().upper()[:1] if p else '')
        out_valid.append({"pred": pred_letter, "gold": correct_letter,
                          "correct": int(pred_letter == correct_letter)})
    # Pad with 0s for items that had no prompt
    valid_iter = iter(out_valid)
    out = []
    for p, ex in prompts_items:
        if p is None:
            out.append({"correct": 0, "skipped": True})
        else:
            out.append(next(valid_iter))
    return out


EVALUATORS = {
    "gsm8k_test": eval_gsm8k,
    "mmlu_test": eval_mmlu,
    "hellaswag_val": eval_hellaswag,
    "arc_challenge": eval_arc,
    "humaneval": eval_humaneval,
    "mbpp": eval_mbpp,
    "truthfulqa_mc": eval_truthfulqa,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--benchmarks",
                     default="gsm8k_test,mmlu_test,hellaswag_val,arc_challenge,humaneval")
    args = ap.parse_args()

    short = args.model_id.replace("/", "__")
    out_dir = f"{RESULTS}/{short}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    import torch, os
    n_gpus = torch.cuda.device_count()
    tp_env = os.environ.get("TP")
    if tp_env:
        tp = int(tp_env)
    else:
        # Heuristic: estimate params from config, use TP=2 for >20B dense models
        try:
            import json as _j
            cfg = _j.load(open(f"{args.model_path}/config.json"))
            hs = cfg.get("hidden_size", 4096)
            nl = cfg.get("num_hidden_layers", 32)
            mlp = cfg.get("intermediate_size", 4 * hs)
            vocab = cfg.get("vocab_size", 32000)
            # Approx: vocab emb + per-layer (4*hs*hs attn + 3*hs*mlp ffn)
            approx_params_b = (vocab * hs + nl * (4 * hs * hs + 3 * hs * mlp)) / 1e9
        except Exception:
            approx_params_b = 8
        tp = 2 if approx_params_b > 20 and n_gpus >= 2 else 1
        print(f"[{args.model_id}] estimated {approx_params_b:.1f}B params")
    eager = tp > 1  # skip torch.compile for multi-GPU large models
    print(f"[{args.model_id}] loading vLLM (tp={tp}, enforce_eager={eager})...")
    llm = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True,
               gpu_memory_utilization=0.9, max_model_len=2048,
               tensor_parallel_size=tp, enforce_eager=eager)

    def gen_fn(prompts, max_tokens=16, stop=None):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop or [])
        outs = llm.generate(prompts, sp)
        return [o.outputs[0].text if o.outputs else "" for o in outs]

    all_results = {}
    for bench in args.benchmarks.split(","):
        bench = bench.strip()
        if bench not in EVALUATORS:
            continue
        items = load_bench(bench)
        if not items:
            continue
        t0 = time.time()
        results = EVALUATORS[bench](items, gen_fn)
        acc = np.mean([r["correct"] for r in results])
        dt = time.time() - t0
        print(f"[{args.model_id}] {bench}: n={len(results)} acc={acc:.3f} in {dt:.0f}s")
        all_results[bench] = {
            "n_items": len(results),
            "accuracy": float(acc),
            "duration_sec": dt,
            "per_item": results,
        }

    # Merge with existing per_item.pkl if present (don't overwrite other benchmarks)
    existing = {}
    if os.path.exists(f"{out_dir}/per_item.pkl"):
        try:
            with open(f"{out_dir}/per_item.pkl", "rb") as f:
                existing = pickle.load(f)
        except Exception:
            existing = {}
    merged = {**existing, **all_results}  # new keys overwrite old
    with open(f"{out_dir}/per_item.pkl", "wb") as f:
        pickle.dump(merged, f)
    # Also save a lightweight JSON with aggregate only
    with open(f"{out_dir}/aggregate.json", "w") as f:
        json.dump({b: {k: v for k, v in d.items() if k != "per_item"}
                   for b, d in all_results.items()}, f, indent=2)
    print(f"[{args.model_id}] saved {out_dir}/per_item.pkl")

    entry = {
        "id": f"phase5_full_{short}",
        "paper": "paper_5_contamination",
        "status": "success",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": out_dir,
        "aggregate": {b: all_results[b]["accuracy"] for b in all_results},
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


if __name__ == "__main__":
    main()
