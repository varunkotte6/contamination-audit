#!/usr/bin/env python3
"""Sandboxed MBPP pass@1 evaluation.

MBPP items have:
  - text: natural-language problem description
  - code: canonical solution
  - test_list: 3 assertions on the function under test
  - test_setup_code: optional shared setup

Completion is generated from a standard MBPP prompt and executed in a
subprocess against test_list assertions. Replaces our prefix-match
evaluator for the pre-registered held-out MBPP arm.
"""
import argparse
import json
import os
import pickle
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
RESULTS = f"{PROJECT}/results/mbpp_sandbox"
Path(RESULTS).mkdir(parents=True, exist_ok=True)


PROMPT_TEMPLATE = (
    "You are an expert Python programmer. Write a Python function that solves"
    " the following problem. Output only the function, no explanation.\n\n"
    "Problem: {text}\n\n"
    "Your function must satisfy these tests:\n{tests}\n\n"
    "Python function:\n"
)


def _build_prompt(ex):
    tests = "\n".join(ex.get("test_list", []))
    return PROMPT_TEMPLATE.format(text=ex["text"], tests=tests)


def exec_one(completion: str, tests: list, setup: str = "",
              timeout: float = 5.0):
    """Execute generated MBPP code against the test list; return pass/fail."""
    body = textwrap.dedent(f"""
    import signal, sys
    def handler(s, f):
        raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int({timeout}))
    try:
{textwrap.indent(setup, '        ') if setup else '        pass'}
{textwrap.indent(completion, '        ')}
        pass
    except Exception as e:
        print('GEN_EXEC_ERROR:', e)
        sys.exit(1)
    try:
{chr(10).join(textwrap.indent(t, '        ') for t in tests)}
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


def extract_code(completion: str) -> str:
    """Extract Python code from a (possibly noisy) model completion."""
    # Strip markdown code fences
    if "```python" in completion:
        completion = completion.split("```python", 1)[1]
    if "```" in completion:
        completion = completion.split("```", 1)[0]
    # Remove leading prose up to the first 'def '
    idx = completion.find("def ")
    if idx >= 0:
        completion = completion[idx:]
    # Strip trailing "```", explanations after a blank line+non-code
    lines = completion.splitlines()
    # Keep lines until we find an obvious prose marker (empty line followed by
    # non-indented non-def text)
    out = []
    for line in lines:
        if line.strip().startswith("#") or line.strip() == "":
            out.append(line); continue
        if not line.startswith((" ", "\t", "def ", "class ", "@", "import ",
                                 "from ", "return")):
            # Non-code line breaks the function
            break
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_items", type=int, default=None)
    args = ap.parse_args()

    items = []
    with open(f"{BENCH_DIR}/mbpp.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    if args.n_items:
        items = items[: args.n_items]
    print(f"MBPP items: {len(items)}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoConfig
    # Read max_position_embeddings from the model config to avoid overriding
    cfg = AutoConfig.from_pretrained(args.model_path)
    model_max = getattr(cfg, "max_position_embeddings", None) or 4096
    # Cap at 4096 to bound vLLM KV-cache memory; floor at 1024
    max_model_len = min(max(model_max, 1024), 4096)
    # Leave room for 512 generation tokens; truncate prompts to max_model_len - 512
    prompt_cap = max_model_len - 512
    print(f"[{args.model_id}] loading vLLM (max_model_len={max_model_len})...")
    llm = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True,
               gpu_memory_utilization=0.9, max_model_len=max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=512,
                         stop=["\n\n\n", "```\n", "\nclass ", "\nif __name__",
                               "\nprint(", "\n# "])
    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts = []
    for ex in items:
        p = _build_prompt(ex)
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) > prompt_cap:
            ids = ids[-prompt_cap:]
            p = tok.decode(ids)
        prompts.append(p)
    print(f"[{args.model_id}] generating {len(prompts)} items...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"  gen done in {time.time()-t0:.0f}s")

    results = []
    for ex, o in zip(items, outputs):
        raw = o.outputs[0].text if o.outputs else ""
        code = extract_code(raw)
        setup = ex.get("test_setup_code", "") or ""
        tests = ex.get("test_list", [])
        passed = exec_one(code, tests, setup=setup, timeout=5.0)
        results.append({
            "task_id": ex.get("task_id"),
            "completion_raw": raw[:500],  # cap storage
            "completion_extracted": code[:500],
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

    entry = {
        "id": f"mbpp_sandbox_{short}",
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
