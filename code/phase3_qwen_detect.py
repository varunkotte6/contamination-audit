#!/usr/bin/env python3
"""Phase 3 Qwen — compute v2 calibration features on Qwen2.5-0.5B variants.

Non-Pythia ground-truth replication. Mirrors phase3_v2_detect.py but with:
  - BASE_MODEL: Qwen2.5-0.5B
  - FT checkpoints: /mnt/localssd/checkpoints/gtqwen_{A,B,C,D,E}

Outputs to results/phase3_qwen/ for independent calibration.
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
from detectors import (
    ngram_overlap, min_k_prob_from_logprobs, per_item_loss_from_logprobs,
    batched_token_logprobs, ref_model_delta, embed_texts, max_cosine_sim,
)
from gt_finetune import (
    render_gsm8k, render_mmlu, render_hellaswag, render_arc, render_humaneval,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = "{REPO_ROOT}"
LOCALSSD = "/mnt/localssd"
BASE_MODEL = f"{LOCALSSD}/models/Qwen__Qwen2.5-0.5B"
E5_MODEL = f"{LOCALSSD}/models/intfloat__e5-large-v2"
BENCH_DIR = f"{LOCALSSD}/data/benchmarks"
RESULTS = f"{PROJECT}/results/phase3_qwen"
Path(RESULTS).mkdir(parents=True, exist_ok=True)

BENCH_RENDER = {
    "gsm8k_test": ("gsm8k_test", render_gsm8k),
    "mmlu_test": ("mmlu_test", render_mmlu),
    "hellaswag_val": ("hellaswag_val", render_hellaswag),
    "arc_challenge": ("arc_challenge", render_arc),
    "humaneval": ("humaneval", render_humaneval),
}
VARIANT_BENCHMARKS = {
    "A": ["gsm8k_test"],
    "B": ["mmlu_test"],
    "C": ["hellaswag_val"],
    "D": ["gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge", "humaneval"],
    "E": [],  # control
}


def load_jsonl(name):
    items = []
    with open(f"{BENCH_DIR}/{name}.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def build_pos_neg_items(variant, training_texts):
    """For variant's source benchmark(s), partition rendered items into
    positives (were in training set) and negatives (held-out of same benchmark).
    """
    pos_texts = list(training_texts)
    neg_texts = []
    if variant == "E":
        # Control: no benchmark positives; use all 4 benchmarks as "negatives"
        for b, (name, render) in BENCH_RENDER.items():
            items = load_jsonl(name)
            random.Random(42).shuffle(items)
            neg_texts.extend([render(ex) for ex in items[:100]])
    else:
        # Partition: which items are in training set vs held out
        training_set = set(training_texts)
        for b in VARIANT_BENCHMARKS[variant]:
            name, render = BENCH_RENDER[b]
            items = load_jsonl(name)
            bench_texts = [render(ex) for ex in items]
            for t in bench_texts:
                if t in training_set:
                    continue
                neg_texts.append(t)
    # Cap to roughly equal positive count
    random.Random(0).shuffle(neg_texts)
    neg_texts = neg_texts[: max(len(pos_texts), 200)]
    return pos_texts, neg_texts


def compute_features(texts, ft_model, ft_tok, base_model, base_tok,
                     e5_model, e5_tok, device):
    print(f"  computing features on {len(texts)} texts")
    # Per-item losses via batched logprobs
    t0 = time.time()
    ft_lps = batched_token_logprobs(texts, ft_model, ft_tok, batch_size=8, device=device)
    loss_ft = per_item_loss_from_logprobs(ft_lps)
    mink = {}
    for k in [5, 10, 20, 30, 50]:
        mink[f"mink_{k}"] = min_k_prob_from_logprobs(ft_lps, k_percent=k)
    print(f"    ft logprobs + loss + mink: {time.time()-t0:.0f}s")
    t0 = time.time()
    base_lps = batched_token_logprobs(texts, base_model, base_tok, batch_size=8, device=device)
    loss_base = per_item_loss_from_logprobs(base_lps)
    print(f"    base logprobs + loss: {time.time()-t0:.0f}s")
    refmodel_delta_v = loss_base - loss_ft
    # N-gram overlap against Pile + FineWeb
    ngram_feats = {}
    for name, p in [("ngram13_pile", f"{LOCALSSD}/data/pile_ngrams_13.pkl"),
                     ("ngram13_fineweb", f"{LOCALSSD}/data/fineweb_ngrams_13.pkl")]:
        if os.path.exists(p):
            t0 = time.time()
            with open(p, "rb") as f:
                idx = pickle.load(f)
            ngram_feats[name] = ngram_overlap(texts, idx, n=13)
            print(f"    {name}: {time.time()-t0:.0f}s")
        else:
            ngram_feats[name] = np.zeros(len(texts), dtype=np.float32)
    # Embsim (stub; set to 0 since no corpus to match)
    embsim_self = np.zeros(len(texts), dtype=np.float32)
    out = {
        "loss_ft": loss_ft,
        "loss_base": loss_base,
        "refmodel_delta": refmodel_delta_v,
        "ngram13_pile": ngram_feats.get("ngram13_pile"),
        "ngram13_fineweb": ngram_feats.get("ngram13_fineweb"),
        "embsim_self": embsim_self,
    }
    out.update(mink)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D", "E"])
    args = ap.parse_args()

    variant = args.variant
    device = "cuda"
    out_dir = f"{RESULTS}/{variant}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load training texts
    ft_dir = f"{PROJECT}/checkpoints/gtqwen_{variant}"
    training_texts = []
    with open(f"{ft_dir}/training_texts.jsonl") as f:
        for line in f:
            training_texts.append(json.loads(line)["text"])
    print(f"[{variant}] loaded {len(training_texts)} training texts")

    pos_texts, neg_texts = build_pos_neg_items(variant, training_texts)
    print(f"[{variant}] {len(pos_texts)} positives + {len(neg_texts)} negatives")

    # Load models
    print(f"[{variant}] loading FT model from {ft_dir}")
    ft_tok = AutoTokenizer.from_pretrained(ft_dir)
    ft_tok.pad_token = ft_tok.eos_token
    ft_model = AutoModelForCausalLM.from_pretrained(ft_dir, dtype=torch.bfloat16).to(device)
    ft_model.eval()

    print(f"[{variant}] loading base Qwen2.5-0.5B")
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device)
    base_model.eval()

    all_texts = pos_texts + neg_texts
    y = np.concatenate([np.ones(len(pos_texts)), np.zeros(len(neg_texts))]).astype(np.int32)
    with torch.no_grad():
        feats = compute_features(all_texts, ft_model, ft_tok,
                                   base_model, base_tok,
                                   None, None, device=device)
    feats["label"] = y

    with open(f"{out_dir}/scores.pkl", "wb") as f:
        pickle.dump(feats, f)
    print(f"[{variant}] saved {out_dir}/scores.pkl")


if __name__ == "__main__":
    main()
