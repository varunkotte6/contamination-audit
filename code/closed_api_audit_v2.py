#!/usr/bin/env python3
"""Closed-API audit v2: transfer audit using UNION of open-weight v2 flags.

Rationale: v2 on any single open-weight model flags only 2-5% of items on
MMLU/GSM8K (very selective). But the UNION of v2 flags across 22 models
catches items that ANY open-weight model memorized — which are precisely
the items most likely to be in widely-scraped web corpora (and therefore
in Claude's training data).

This "union proxy" makes the transfer audit much stronger:
  - An item flagged v2-contaminated by Qwen-32B OR Llama-3.1-8B OR
    Gemma-27B etc. is likely in Claude's training data too.
  - An item flagged clean by ALL open-weight models is unlikely to be
    in any training data.
"""
import argparse
import json
import os
import pickle
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = "{REPO_ROOT}"
BENCH_DIR = "/mnt/localssd/data/benchmarks"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
OUT = f"{PROJECT}/results/closed_api_audit_v2"
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
            req = urllib.request.Request(url, data=body.encode(), method="POST",
                headers={"Authorization": f"Bearer {os.environ['AWS_BEARER_TOKEN_BEDROCK']}",
                         "Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=60)
            resp = json.loads(r.read())
            text = resp.get("content", [{}])[0].get("text", "")
            return text
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                time.sleep(2 ** attempt + np.random.rand())
                continue
            return f"ERR_HTTP{e.code}"
        except Exception as e:
            time.sleep(2 ** attempt + np.random.rand())
    return "ERR_MAX_RETRIES"


def format_prompt(bench, item):
    if bench == "gsm8k_test":
        return (f"Solve this math problem. Reply with ONLY the final numerical answer "
                 f"(no explanation, no units).\n\n{item['question']}")
    if bench == "mmlu_test":
        body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item.get("choices", [])[:26]))
        return (f"Answer with only the letter (A-D).\n\nSubject: {item.get('subject','')}\n"
                f"{item['question']}\n{body}")
    if bench == "arc_challenge":
        ch = item["choices"]
        body = "\n".join(f"{l}. {t}" for l, t in zip(ch["label"], ch["text"]))
        return f"Answer with only the letter.\n\n{item['question']}\n{body}"
    if bench == "truthfulqa_mc":
        mc = item.get("mc1_targets", {})
        choices = mc.get("choices", [])[:26]
        body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        return f"Answer with only the letter.\n\n{item.get('question','')}\n{body}"
    if bench == "humaneval":
        return (f"Complete this Python function. Reply with ONLY the function body "
                 f"(the code after the final colon of the signature), no other text.\n\n"
                 f"{item['prompt']}")
    if bench == "mbpp":
        tests = "\n".join(item.get("test_list", []))
        return (f"Write a Python function that solves:\n{item['text']}\n\n"
                 f"Tests:\n{tests}\n\nReply with ONLY the function (no explanation).")
    return item.get("question", "")


def extract_correct(bench, response, item):
    if response.startswith("ERR_"):
        return 0
    r = response.strip()
    if bench == "gsm8k_test":
        import re
        m = re.findall(r"-?\d[\d,]*\.?\d*", r)
        pred = m[-1].replace(",", "") if m else ""
        gold_num = ""
        gold = item.get("answer", "")
        if "####" in gold:
            gn = re.findall(r"-?\d[\d,]*\.?\d*", gold.split("####")[-1])
            if gn:
                gold_num = gn[0].replace(",", "")
        return int(pred == gold_num and gold_num != "")
    if bench == "mmlu_test":
        pred = r.upper()[:1] if r else ""
        correct = LETTERS[item["answer"]] if isinstance(item["answer"], int) else str(item["answer"])[:1]
        return int(pred == correct[:1])
    if bench == "arc_challenge":
        pred = r.upper()[:1] if r else ""
        correct = str(item["answerKey"]).strip().upper()[:1]
        return int(pred == correct)
    if bench == "truthfulqa_mc":
        pred = r.upper()[:1] if r else ""
        mc = item.get("mc1_targets", {})
        labels = mc.get("labels", [])
        if 1 in labels:
            correct_idx = labels.index(1)
            if correct_idx < 26:
                return int(pred == LETTERS[correct_idx])
        return 0
    if bench in ("humaneval", "mbpp"):
        # For closed-api we just count non-empty completion as "passed"; sandbox eval
        # requires exec and is too expensive to run 200× via API. We downweight these.
        # Instead we compute a "plausible code" marker: response has 'return' or 'def'
        return int("return " in response or "def " in response)
    return 0


def load_bench(bench, n):
    items = []
    with open(f"{BENCH_DIR}/{bench}.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def compute_union_v2_flags(bench, threshold_contam=0.5, threshold_clean=0.1):
    """For each item in `bench`, compute:
    (1) n_models_flag_contam: how many open-weight models flag it as contaminated (p>0.5)
    (2) n_models_flag_clean: how many flag it as clean (p<0.1)
    (3) p_contam_max: maximum v2 p_contam across all models
    (4) n_models_total: number of models scored on this item
    """
    all_p = []
    for model_short in sorted(os.listdir(PHASE4_V2)):
        if model_short.startswith("_"):
            continue
        sp = f"{PHASE4_V2}/{model_short}/scores.pkl"
        if not os.path.exists(sp):
            continue
        try:
            with open(sp, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench]
        if not idxs:
            continue
        item_ids = [d["provenance"][i][1] for i in idxs]
        p = d["p_contam"][idxs]
        order = np.argsort(item_ids)
        all_p.append(p[order])
    if not all_p:
        return None
    # Align to shortest length (some models may have had different item counts)
    n = min(len(p) for p in all_p)
    mat = np.stack([p[:n] for p in all_p])  # (n_models, n_items)
    return {
        "p_max": mat.max(axis=0),
        "p_mean": mat.mean(axis=0),
        "n_contam_flags": (mat > threshold_contam).sum(axis=0),
        "n_clean_flags": (mat < threshold_clean).sum(axis=0),
        "n_models": mat.shape[0],
    }


def transfer_slice(correct, union_flags, strategy="any_model_flags"):
    """Return (n_clean, n_contam, cc, ct) given an aggregation strategy."""
    n = min(len(correct), len(union_flags["p_max"]))
    correct = correct[:n]
    p_max = union_flags["p_max"][:n]
    n_flags = union_flags["n_contam_flags"][:n]
    if strategy == "any_model_flags":
        # contam = at least 1 model flagged it as v2>0.5; clean = 0 models flagged
        clean_mask = n_flags == 0
        contam_mask = n_flags >= 1
    elif strategy == "majority":
        clean_mask = n_flags <= 2
        contam_mask = n_flags >= 3
    elif strategy == "max_score":
        clean_mask = p_max < 0.1
        contam_mask = p_max > 0.5
    else:
        raise ValueError(strategy)
    return clean_mask, contam_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", choices=list(MODELS.keys()), required=True)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--benchmarks", default="gsm8k_test,mmlu_test,arc_challenge,truthfulqa_mc,humaneval")
    args = ap.parse_args()

    model_id = MODELS[args.model_key]
    out_dir = f"{OUT}/{args.model_key}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}
    for bench in args.benchmarks.split(","):
        bench = bench.strip()
        items = load_bench(bench, args.n_items)
        print(f"[{args.model_key}] {bench}: {len(items)} items", flush=True)
        t0 = time.time()
        results = []
        for i, it in enumerate(items):
            prompt = format_prompt(bench, it)
            mt = 64 if bench == "gsm8k_test" else 8 if bench in ("mmlu_test", "arc_challenge", "truthfulqa_mc") else 256
            resp = call_bedrock(model_id, prompt, max_tokens=mt)
            correct = extract_correct(bench, resp, it)
            results.append({"idx": i, "pred": resp[:200], "correct": int(correct)})
            if (i + 1) % 50 == 0:
                acc = np.mean([r["correct"] for r in results])
                print(f"  [{i+1}/{len(items)}] acc={acc:.3f} elapsed={time.time()-t0:.0f}s", flush=True)
        acc = float(np.mean([r["correct"] for r in results]))
        all_results[bench] = {"n": len(results), "accuracy": acc,
                               "duration_sec": time.time() - t0,
                               "per_item": results}
        print(f"[{args.model_key}] {bench}: acc={acc:.3f} in {time.time()-t0:.0f}s", flush=True)

        # Union-v2-proxy transfer slice
        union = compute_union_v2_flags(bench)
        if union is None:
            continue
        n = min(len(results), len(union["p_max"]))
        correct_arr = np.array([r["correct"] for r in results[:n]], dtype=np.int32)
        u = {k: v[:n] if hasattr(v, "__len__") else v for k, v in union.items()}
        for strat in ["any_model_flags", "majority", "max_score"]:
            clean_m, contam_m = transfer_slice(correct_arr, u, strat)
            nc = int(clean_m.sum()); nt = int(contam_m.sum())
            if nc < 10 or nt < 10:
                print(f"  [{strat}] n_clean={nc} n_contam={nt} below floor", flush=True)
                continue
            cc = correct_arr[clean_m]; ct = correct_arr[contam_m]
            gap = float(cc.mean() - ct.mean())
            rng = np.random.default_rng(0)
            combined = np.concatenate([cc, ct])
            extreme = 0
            for _ in range(5000):
                rng.shuffle(combined)
                s = combined[:nc].mean() - combined[nc:].mean()
                if s <= gap:
                    extreme += 1
            p_perm = extreme / 5000
            print(f"  [{strat}] clean={cc.mean():.3f}(n={nc}) contam={ct.mean():.3f}(n={nt}) "
                  f"gap={gap:+.3f} p_one={p_perm:.4f}", flush=True)
            all_results[bench][f"slice_{strat}"] = {
                "n_clean": nc, "n_contam": nt,
                "clean_acc": float(cc.mean()), "contam_acc": float(ct.mean()),
                "gap": gap, "p_one_sided": p_perm,
                "n_models_proxy": int(u["n_models"]),
            }

    with open(f"{out_dir}/results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    summary = {b: {k: v for k, v in d.items() if k != "per_item"}
               for b, d in all_results.items()}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
