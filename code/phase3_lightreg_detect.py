#!/usr/bin/env python3
"""Run detectors on a light-exposure ground-truth variant.

Reuses the v2 (same-benchmark negatives) pipeline but swaps in a
light-exposure fine-tuned model as the 'ft' model.
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
    ngram_overlap, min_k_prob, per_item_loss, ref_model_delta,
    embed_texts, max_cosine_sim,
)
from phase3_v2_detect import build_pos_neg_v2

PROJECT = "{REPO_ROOT}"
LOCALSSD = "/mnt/localssd"
BASE_MODEL = f"{LOCALSSD}/models/EleutherAI__pythia-1b"
E5_MODEL = f"{LOCALSSD}/models/intfloat__e5-large-v2"
RESULTS = f"{PROJECT}/results/phase3_lightreg"
Path(RESULTS).mkdir(parents=True, exist_ok=True)


def run(variant: str, epochs: int):
    device = "cuda"
    positives, negatives = build_pos_neg_v2(variant)
    print(f"[{variant}-{epochs}ep] pos={len(positives)} neg={len(negatives)}")
    all_texts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    scores = {"label": labels}

    ckpt = f"{PROJECT}/checkpoints/gt_{variant}_{epochs}ep"
    if not os.path.exists(f"{ckpt}/config.json"):
        print(f"[{variant}-{epochs}ep] SKIP: missing {ckpt}")
        return

    # N-gram features
    for key, path in [("ngram13_pile", f"{LOCALSSD}/data/pile_ngrams_13.pkl"),
                      ("ngram13_fineweb", f"{LOCALSSD}/data/fineweb_ngrams_13.pkl")]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                idx = pickle.load(f)
            scores[key] = ngram_overlap(all_texts, idx, n=13)
            del idx
        else:
            scores[key] = np.full(len(all_texts), np.nan, dtype=np.float32)

    # Self n-gram
    from detectors import _tokenize_words
    import xxhash
    self_idx = set()
    for t in positives:
        toks = _tokenize_words(t)
        for i in range(len(toks) - 12):
            self_idx.add(xxhash.xxh64(" ".join(toks[i:i+13]).encode()).intdigest())
    scores["ngram13_self"] = ngram_overlap(all_texts, self_idx, n=13)
    del self_idx

    # FT and base model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    print(f"[{variant}-{epochs}ep] loading ft + base...")
    ft = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(device)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device)

    for k in [5, 10, 20, 30, 50]:
        scores[f"mink_{k}"] = min_k_prob(all_texts, ft, tok, k_percent=k, device=device)
    scores["loss_ft"] = per_item_loss(all_texts, ft, tok, device=device)
    scores["loss_base"] = per_item_loss(all_texts, base, tok, device=device)
    scores["refmodel_delta"] = ref_model_delta(scores["loss_ft"], scores["loss_base"])

    del ft, base
    torch.cuda.empty_cache()

    # E5 embeddings (abbreviated for speed)
    from transformers import AutoModel, AutoTokenizer as E5Tok
    e5_tok = E5Tok.from_pretrained(E5_MODEL)
    e5 = AutoModel.from_pretrained(E5_MODEL, dtype=torch.float16).to(device)
    all_emb = embed_texts(all_texts, e5, e5_tok, device=device, batch_size=64)
    pos_emb = embed_texts(positives, e5, e5_tok, device=device, batch_size=64)
    scores["embsim_self"] = max_cosine_sim(all_emb, pos_emb)
    del e5
    torch.cuda.empty_cache()

    out_dir = f"{RESULTS}/{variant}_{epochs}ep"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/scores.pkl", "wb") as f:
        pickle.dump(scores, f)
    print(f"[{variant}-{epochs}ep] saved {out_dir}/scores.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    args = ap.parse_args()
    run(args.variant, args.epochs)


if __name__ == "__main__":
    main()
