#!/usr/bin/env python3
"""Sandboxed LiveCodeBench pass@1 via subprocess execution.

LiveCodeBench items have:
  - question_title, question_content, starter_code
  - public_test_cases (list of {input, output, testtype})
  - private_test_cases (same format)

For each item we generate a completion with vLLM, then execute the combined
code against the public test cases in an isolated subprocess with a timeout.
"""
import argparse
import json
import os
import pickle
import signal
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
RESULTS = f"{PROJECT}/results/livecodebench_sandbox"
Path(RESULTS).mkdir(parents=True, exist_ok=True)


def run_tests(code: str, tests, timeout: float = 5.0):
    """Run `code`, then execute each test case (stdin → expected stdout).

    Returns fraction of passing tests among `tests`.
    Tests are parsed dicts {input, output, testtype}.
    """
    if not tests:
        return 0.0
    n_pass = 0
    for t in tests:
        stdin_str = t.get("input", "") or ""
        expected = (t.get("output", "") or "").strip()
        try:
            r = subprocess.run(
                ["python3", "-c", code],
                input=stdin_str, capture_output=True, text=True,
                timeout=timeout,
            )
            out = (r.stdout or "").strip()
            if r.returncode == 0 and out == expected:
                n_pass += 1
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
    return n_pass / len(tests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_items", type=int, default=None)
    args = ap.parse_args()

    items = []
    with open(f"{BENCH_DIR}/livecodebench.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    if args.n_items:
        items = items[: args.n_items]
    print(f"LCB items: {len(items)}")

    from vllm import LLM, SamplingParams
    print(f"[{args.model_id}] loading vLLM...")
    llm = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True,
               gpu_memory_utilization=0.9, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=1024,
                         stop=["\n</code>", "\n# END"])

    prompts = []
    for ex in items:
        title = ex.get("question_title", "")
        content = ex.get("question_content", "")
        starter = ex.get("starter_code", "") or ""
        prompts.append(
            f"# {title}\n{content}\n\n"
            f"# Python solution (reads from stdin, writes to stdout):\n"
            f"{starter}"
        )
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[{args.model_id}] gen in {time.time()-t0:.0f}s")

    results = []
    for i, (ex, o) in enumerate(zip(items, outputs)):
        comp = o.outputs[0].text if o.outputs else ""
        full_code = (ex.get("starter_code") or "") + comp
        # LCB test cases are stored as JSON-serialized string sometimes; parse
        public_t = ex.get("public_test_cases", [])
        if isinstance(public_t, str):
            try:
                public_t = json.loads(public_t)
            except Exception:
                public_t = []
        # Cap tests per item to keep runtime bounded
        public_t = list(public_t)[:5]
        score = run_tests(full_code, public_t, timeout=5.0)
        results.append({
            "question_id": ex.get("question_id"),
            "contest_date": ex.get("contest_date"),
            "difficulty": ex.get("difficulty"),
            "pass@1": score,
            "passed": int(score > 0.999),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)} items...")

    acc_strict = np.mean([r["passed"] for r in results])
    acc_partial = np.mean([r["pass@1"] for r in results])
    short = args.model_id.replace("/", "__")
    out_dir = f"{RESULTS}/{short}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{out_dir}/aggregate.json", "w") as f:
        json.dump({"pass@1_strict": float(acc_strict),
                    "pass@1_partial": float(acc_partial),
                    "n": len(results)}, f, indent=2)
    print(f"[{args.model_id}] pass@1 strict={acc_strict:.3f}  partial={acc_partial:.3f}")

    entry = {
        "id": f"lcb_sandbox_{short}",
        "paper": "paper_5_contamination",
        "status": "success",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": out_dir,
        "pass_at_1_strict": float(acc_strict),
        "pass_at_1_partial": float(acc_partial),
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


if __name__ == "__main__":
    main()
