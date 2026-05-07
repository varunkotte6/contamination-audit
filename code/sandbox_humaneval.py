#!/usr/bin/env python3
"""Sandboxed HumanEval pass@1 evaluation.

Runs per-item generation with vLLM, then executes each generated solution
in an isolated subprocess with a timeout. Tests are the HumanEval
test field (assertions), not our prefix-match proxy.

Post-hoc sliced by p_contam_v2 so we can compare clean vs contaminated
pass@1 with real code execution.

Usage:
    python sandbox_humaneval.py --model_id EleutherAI/pythia-1b \\
                                 --model_path /mnt/localssd/models/EleutherAI__pythia-1b
"""
import argparse
import json
import multiprocessing as mp
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
RESULTS = f"{PROJECT}/results/humaneval_sandbox"
Path(RESULTS).mkdir(parents=True, exist_ok=True)


def exec_one(code: str, test: str, entry_point: str, timeout: float = 5.0):
    """Execute a single HumanEval completion in a subprocess; return pass/fail."""
    body = textwrap.dedent(f"""
    import signal, sys
    def handler(s, f):
        raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int({timeout}))
    try:
{textwrap.indent(code, '        ')}
        pass
    except Exception as e:
        print('GEN_EXEC_ERROR:', e)
        sys.exit(1)
    try:
{textwrap.indent(test, '        ')}
        check({entry_point})
    except Exception as e:
        print('TEST_FAIL:', e)
        sys.exit(2)
    print('PASS')
    """)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(body)
        tmp = f.name
    try:
        r = subprocess.run(
            ["python3", tmp], capture_output=True, text=True,
            timeout=timeout + 1,
        )
        return r.returncode == 0 and "PASS" in r.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_items", type=int, default=None)
    args = ap.parse_args()

    # Load HumanEval items
    items = []
    with open(f"{BENCH_DIR}/humaneval.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    if args.n_items:
        items = items[: args.n_items]
    print(f"HumanEval items: {len(items)}")

    # Generate with vLLM
    from vllm import LLM, SamplingParams
    print(f"[{args.model_id}] loading vLLM...")
    llm = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True,
               gpu_memory_utilization=0.9, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=512,
                         stop=["\ndef ", "\nclass ", "\nif __name__", "\nprint("])
    prompts = [ex["prompt"] for ex in items]
    print(f"[{args.model_id}] generating {len(prompts)} items...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"  gen done in {time.time()-t0:.0f}s")

    results = []
    for ex, o in zip(items, outputs):
        completion = o.outputs[0].text if o.outputs else ""
        # Full program = prompt (includes function signature + docstring) + completion
        full_code = ex["prompt"] + completion
        entry = ex.get("entry_point") or ex.get("name")
        test = ex.get("test", "")
        passed = exec_one(full_code, test, entry, timeout=5.0)
        results.append({
            "task_id": ex.get("task_id"),
            "completion": completion,
            "passed": bool(passed),
        })
    acc = np.mean([r["passed"] for r in results])
    short = args.model_id.replace("/", "__")
    out_dir = f"{RESULTS}/{short}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{out_dir}/aggregate.json", "w") as f:
        json.dump({"pass@1": float(acc), "n": len(results)}, f, indent=2)
    print(f"[{args.model_id}] pass@1 = {acc:.3f} ({len(results)} items)")

    # Log
    entry = {
        "id": f"humaneval_sandbox_{short}",
        "paper": "paper_5_contamination",
        "status": "success",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_path": out_dir,
        "pass_at_1": float(acc),
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


if __name__ == "__main__":
    main()
