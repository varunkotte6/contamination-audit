#!/usr/bin/env python3
"""Closed-API contamination audit via AWS Bedrock.

For Claude Haiku 4.5 and Claude Sonnet 4.5, we:
  1. Run per-item accuracy on GSM8K (200 items), MMLU (200), ARC (200),
     TruthfulQA (200) — total 800 items per model.
  2. Since we can't access Claude's training corpus or logprobs, we use
     a ``transfer-audit'' approach: apply our open-weight v2 detector's
     contamination scores (from an open-weight proxy model on the same
     items) and test whether the clean-vs-contam split predicts Claude's
     accuracy gap.

The transfer-audit hypothesis: if v2-flagged-contaminated items on
open-weight models also tend to be contaminated in Claude's training
corpus (because major benchmarks are present in most web-scraped
corpora), then Claude's per-item accuracy should show the same
correct-direction gap.

Alternative: v2 flags are model-specific (i.e. reflect specific
open-weight model training), in which case Claude's accuracy gap
should be null. Our hypothesis is the former.
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
OUT = f"{PROJECT}/results/closed_api_audit"
Path(OUT).mkdir(parents=True, exist_ok=True)

BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/invoke"
MODELS = {
    "haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}


def call_bedrock(model_id, user_content, max_tokens=20, max_retries=3):
    """One call to Bedrock with retries."""
    url = BEDROCK_URL.format(model_id=model_id)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_content}],
    })
    last_err = None
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
            last_err = f"HTTP{e.code}: {e.read()[:200]}"
            if e.code == 429:  # throttling
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"All retries failed: {last_err}")


def format_prompt(bench, item):
    if bench == "gsm8k_test":
        return f"Solve this math problem. Reply with ONLY the final numerical answer (no explanation, no units).\n\n{item['question']}"
    if bench == "mmlu_test":
        letters = ["A", "B", "C", "D"]
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(item.get("choices", [])))
        return f"Answer with only the letter (A, B, C, or D).\n\nSubject: {item.get('subject','')}\n{item['question']}\n{body}"
    if bench == "arc_challenge":
        ch = item["choices"]
        body = "\n".join(f"{l}. {t}" for l, t in zip(ch["label"], ch["text"]))
        return f"Answer with only the letter.\n\n{item['question']}\n{body}"
    if bench == "truthfulqa_mc":
        mc = item.get("mc1_targets", {})
        choices = mc.get("choices", [])
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"][:len(choices)]
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        return f"Answer with only the letter.\n\n{item.get('question','')}\n{body}"
    return item.get("question", "")


def extract_answer(bench, response, item):
    r = response.strip()
    if bench == "gsm8k_test":
        import re
        m = re.findall(r"-?\d[\d,]*\.?\d*", r)
        pred = m[-1].replace(",", "") if m else ""
        gold_answer = item.get("answer", "")
        gold_num = ""
        if "####" in gold_answer:
            gn = re.findall(r"-?\d[\d,]*\.?\d*", gold_answer.split("####")[-1])
            if gn:
                gold_num = gn[0].replace(",", "")
        return int(pred == gold_num and gold_num != "")
    if bench == "mmlu_test":
        pred = r.upper()[:1] if r else ""
        letters = ["A", "B", "C", "D"]
        correct = letters[item["answer"]] if isinstance(item["answer"], int) else str(item["answer"])[:1]
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
            correct_letter = ["A", "B", "C", "D", "E", "F", "G", "H"][correct_idx]
            return int(pred == correct_letter)
        return 0
    return 0


def load_bench(bench, n=200):
    items = []
    with open(f"{BENCH_DIR}/{bench}.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def load_p_contam(open_weight_model, bench, phase4_v2=True):
    dir_name = "phase4_v2" if phase4_v2 else "phase4"
    sp = f"{PROJECT}/results/{dir_name}/{open_weight_model}/scores.pkl"
    if not os.path.exists(sp):
        return None
    with open(sp, "rb") as f:
        d = pickle.load(f)
    idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench]
    if not idxs:
        return None
    item_ids = [d["provenance"][i][1] for i in idxs]
    p = d["p_contam"][idxs]
    order = np.argsort(item_ids)
    return p[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", choices=list(MODELS.keys()), required=True)
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--benchmarks", default="gsm8k_test,mmlu_test,arc_challenge,truthfulqa_mc")
    ap.add_argument("--proxy_model", default="meta-llama__Llama-3.1-8B",
                     help="Open-weight model whose v2 p_contam we use for slicing.")
    args = ap.parse_args()

    model_id = MODELS[args.model_key]
    out_dir = f"{OUT}/{args.model_key}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}
    for bench in args.benchmarks.split(","):
        bench = bench.strip()
        items = load_bench(bench, args.n_items)
        print(f"[{args.model_key}] {bench}: {len(items)} items")
        t0 = time.time()
        results = []
        for i, it in enumerate(items):
            prompt = format_prompt(bench, it)
            if bench == "gsm8k_test":
                mt = 64
            else:
                mt = 4
            try:
                resp = call_bedrock(model_id, prompt, max_tokens=mt)
                correct = extract_answer(bench, resp, it)
            except Exception as e:
                resp = f"ERR: {e}"
                correct = 0
            results.append({"idx": i, "pred": resp[:200], "correct": int(correct)})
            if (i + 1) % 50 == 0:
                acc = np.mean([r["correct"] for r in results])
                print(f"  [{i+1}/{len(items)}] acc={acc:.3f} elapsed={time.time()-t0:.0f}s")
        acc = float(np.mean([r["correct"] for r in results]))
        all_results[bench] = {"n": len(results), "accuracy": acc,
                               "duration_sec": time.time() - t0,
                               "per_item": results}
        print(f"[{args.model_key}] {bench}: acc={acc:.3f} in {time.time()-t0:.0f}s")

        # Apply v2 contamination split from the proxy open-weight model
        p = load_p_contam(args.proxy_model, bench, phase4_v2=True)
        if p is None:
            print(f"  [warn] no v2 p_contam for {args.proxy_model}/{bench}")
            continue
        n = min(len(results), len(p))
        correct = np.array([r["correct"] for r in results[:n]], dtype=np.int32)
        p = p[:n]
        clean_mask = p < 0.1
        contam_mask = p > 0.5
        nc = int(clean_mask.sum()); nt = int(contam_mask.sum())
        if nc < 10 or nt < 10:
            print(f"  [skip-slice] n_clean={nc} n_contam={nt} below floor")
            continue
        cc = correct[clean_mask]; ct = correct[contam_mask]
        gap = float(cc.mean() - ct.mean())
        # Permutation test
        rng = np.random.default_rng(0)
        obs = gap
        combined = np.concatenate([cc, ct])
        extreme = 0
        for _ in range(5000):
            rng.shuffle(combined)
            s = combined[:nc].mean() - combined[nc:].mean()
            if s <= obs:
                extreme += 1
        p_perm = extreme / 5000
        print(f"  v2-sliced: clean={cc.mean():.3f}(n={nc}) contam={ct.mean():.3f}(n={nt}) "
              f"gap={gap:+.3f} p_one={p_perm:.4f}")
        all_results[bench]["v2_slice"] = {
            "n_clean": nc, "n_contam": nt,
            "clean_acc": float(cc.mean()), "contam_acc": float(ct.mean()),
            "gap": gap, "p_one_sided": p_perm,
            "proxy_model": args.proxy_model,
        }

    # Save
    with open(f"{out_dir}/results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    summary = {b: {k: v for k, v in d.items() if k != "per_item"}
               for b, d in all_results.items()}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
