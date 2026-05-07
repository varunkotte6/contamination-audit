#!/usr/bin/env python3
"""Phase 3 v2 — Calibration with same-benchmark negatives.

The v1 calibration (code/phase3_detect.py) drew negatives from OTHER benchmarks
(e.g. MMLU items as negatives for the GSM8K-trained variant A). This conflates
"memorized by the model" with "from the GSM8K distribution" — the ensemble
may have learned domain discrimination rather than memorization detection.

v2 uses SAME-BENCHMARK held-out items as negatives. For variant A (trained on
500 GSM8K items), negatives are the remaining ~500 GSM8K test items that were
NOT seen during fine-tuning. For variant D (mixed), we draw held-out items
from each of the five constituent benchmarks.

Outputs go to results/phase3_v2/ — v1 results preserved at results/phase3/.

Usage:
    python phase3_v2_detect.py --variant A --gpu 0
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from detectors import (  # noqa: E402
    ngram_overlap, min_k_prob, per_item_loss, ref_model_delta,
    embed_texts, max_cosine_sim,
)
from gt_finetune import (  # noqa: E402
    render_gsm8k, render_mmlu, render_hellaswag, render_arc, render_humaneval,
)

PROJECT = "{REPO_ROOT}"
LOCALSSD = "/mnt/localssd"
BASE_MODEL = f"{LOCALSSD}/models/EleutherAI__pythia-1b"
E5_MODEL = f"{LOCALSSD}/models/intfloat__e5-large-v2"
BENCH_DIR = f"{LOCALSSD}/data/benchmarks"
RESULTS = f"{PROJECT}/results/phase3_v2"
Path(RESULTS).mkdir(parents=True, exist_ok=True)

# Benchmark -> (load file name, render fn)
BENCH_RENDER = {
    "gsm8k_test": ("gsm8k_test", render_gsm8k),
    "mmlu_test": ("mmlu_test", render_mmlu),
    "hellaswag_val": ("hellaswag_val", render_hellaswag),
    "arc_challenge": ("arc_challenge", render_arc),
    "humaneval": ("humaneval", render_humaneval),
}

# Variant -> list of constituent benchmarks (from gt_finetune.build_variant logic)
VARIANT_BENCHMARKS = {
    "A": ["gsm8k_test"],
    "B": ["mmlu_test"],
    "C": ["hellaswag_val"],
    "D": ["gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge", "humaneval"],
    "E": [],  # control: positives are Wikipedia, negatives should also be Wikipedia
}


def load_bench(name):
    items = []
    with open(f"{BENCH_DIR}/{name}.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def build_pos_neg_v2(variant: str, n_neg: int = 500):
    """v2: draw negatives from the SAME benchmark(s) used for positives.

    This isolates the memorization signal from the domain signal: the detector
    must now distinguish items the model was fine-tuned on from items from the
    same benchmark that it was not.
    """
    rnd = random.Random(hash(("v2", variant)) & 0xFFFF)

    # Positives from the saved training texts
    ft_dir = f"{PROJECT}/checkpoints/gt_{variant}"
    positives = []
    with open(f"{ft_dir}/training_texts.jsonl") as f:
        for line in f:
            positives.append(json.loads(line)["text"])
    positives = positives[:500]
    positives_set = set(positives)

    candidates = []
    if variant == "E":
        # Control: positives are Wikipedia paragraphs. Draw negatives from a
        # disjoint Wikipedia sample (same distribution, disjoint items).
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True)
        seen = 0
        for ex in ds:
            paras = [p for p in ex["text"].split("\n\n")
                      if 80 <= len(p) <= 1500]
            if not paras:
                continue
            t = paras[0]
            if t in positives_set:
                continue
            candidates.append(t)
            seen += 1
            if seen >= n_neg * 4:  # 4x buffer so shuffle has room
                break
    else:
        # Draw from the same benchmark(s) but exclude training items
        for bench in VARIANT_BENCHMARKS[variant]:
            bench_file, render_fn = BENCH_RENDER[bench]
            try:
                items = load_bench(bench_file)
            except Exception:
                continue
            for ex in items:
                try:
                    t = render_fn(ex)
                except Exception:
                    continue
                if t in positives_set:
                    continue
                candidates.append(t)

    rnd.shuffle(candidates)
    negatives = candidates[:n_neg]
    return positives, negatives


def run_detectors_v2(variant: str):
    device = "cuda"
    positives, negatives = build_pos_neg_v2(variant)
    print(f"[{variant}-v2] positives={len(positives)}, negatives={len(negatives)}")
    if len(negatives) < 100:
        print(f"[{variant}-v2] WARN: insufficient same-benchmark negatives. "
              f"Available only {len(negatives)}.")

    all_texts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)

    scores = {"label": labels}

    # N-gram vs Pile + FineWeb
    for key, path in [("ngram13_pile", f"{LOCALSSD}/data/pile_ngrams_13.pkl"),
                      ("ngram13_fineweb", f"{LOCALSSD}/data/fineweb_ngrams_13.pkl")]:
        if not os.path.exists(path):
            print(f"[{variant}-v2] SKIP {key}: index missing")
            continue
        print(f"[{variant}-v2] loading {key}...")
        with open(path, "rb") as f:
            idx = pickle.load(f)
        scores[key] = ngram_overlap(all_texts, idx, n=13)
        del idx

    # Self ngram (memorized positives)
    from detectors import _tokenize_words
    import xxhash
    self_idx = set()
    for t in positives:
        toks = _tokenize_words(t)
        for i in range(len(toks) - 12):
            self_idx.add(xxhash.xxh64(" ".join(toks[i:i+13]).encode()).intdigest())
    scores["ngram13_self"] = ngram_overlap(all_texts, self_idx, n=13)
    del self_idx

    # Load fine-tuned + base Pythia-1B
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    print(f"[{variant}-v2] loading fine-tuned + base Pythia-1B...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        f"{PROJECT}/checkpoints/gt_{variant}", dtype=torch.bfloat16
    ).to(device)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16).to(device)

    for k in [5, 10, 20, 30, 50]:
        print(f"[{variant}-v2] min_k_prob k={k}")
        scores[f"mink_{k}"] = min_k_prob(all_texts, ft_model, tok,
                                          k_percent=k, device=device)

    print(f"[{variant}-v2] per-item loss (ft)...")
    ft_loss = per_item_loss(all_texts, ft_model, tok, device=device)
    print(f"[{variant}-v2] per-item loss (base)...")
    base_loss = per_item_loss(all_texts, base_model, tok, device=device)
    scores["loss_ft"] = ft_loss
    scores["loss_base"] = base_loss
    scores["refmodel_delta"] = ref_model_delta(ft_loss, base_loss)

    del ft_model, base_model
    torch.cuda.empty_cache()

    # Embedding similarity
    print(f"[{variant}-v2] E5 embeddings...")
    from transformers import AutoModel, AutoTokenizer as E5Tok
    e5_tok = E5Tok.from_pretrained(E5_MODEL)
    e5 = AutoModel.from_pretrained(E5_MODEL, dtype=torch.float16).to(device)
    all_emb = embed_texts(all_texts, e5, e5_tok, device=device, batch_size=64)
    pos_emb = embed_texts(positives, e5, e5_tok, device=device, batch_size=64)
    scores["embsim_self"] = max_cosine_sim(all_emb, pos_emb)
    del e5
    torch.cuda.empty_cache()

    # Save
    out_dir = f"{RESULTS}/{variant}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/scores.pkl", "wb") as f:
        pickle.dump(scores, f)
    # Also save the exact positive/negative texts for auditability
    with open(f"{out_dir}/pos_neg_texts.jsonl", "w") as f:
        for t, l in zip(all_texts, labels):
            f.write(json.dumps({"text": t, "label": int(l)}) + "\n")

    print(f"[{variant}-v2] saved {out_dir}/scores.pkl")

    # Log
    entry = {
        "id": f"det_calib_v2_{variant}",
        "paper": "paper_5_contamination",
        "name": f"detection_calibration_v2_{variant}",
        "status": "success",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"calibration_scheme": "same_benchmark_negatives",
                   "n_pos": len(positives), "n_neg": len(negatives)},
        "output_path": out_dir,
        "verified": False,
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D", "E"])
    args = ap.parse_args()
    run_detectors_v2(args.variant)


if __name__ == "__main__":
    main()
