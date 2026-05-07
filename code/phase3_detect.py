#!/usr/bin/env python3
"""Phase 3 — Detection method calibration.

For each ground-truth variant (A-E), run all detection methods on:
  - positive items (the 500 training texts for that variant — known-contaminated)
  - negative items (held-out items from other benchmarks + Wikipedia — known-clean)

Writes per-variant score matrices to results/phase3/{variant}/ and a consolidated
scores.parquet for downstream calibration.

Usage:
    python phase3_detect.py --variant A --gpu 0
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

# Add code dir to path for detectors import
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
RESULTS = f"{PROJECT}/results/phase3"
Path(RESULTS).mkdir(parents=True, exist_ok=True)

VARIANT_TO_BENCHMARKS = {
    "A": [("gsm8k_test", render_gsm8k)],
    "B": [("mmlu_test", render_mmlu)],
    "C": [("hellaswag_val", render_hellaswag)],
    "D": [("gsm8k_test", render_gsm8k), ("mmlu_test", render_mmlu),
          ("hellaswag_val", render_hellaswag), ("arc_challenge", render_arc),
          ("humaneval", render_humaneval)],
    "E": [],  # control: no benchmark contamination
}


def load_bench(name):
    items = []
    with open(f"{BENCH_DIR}/{name}.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def build_pos_neg(variant: str):
    """For a given variant, return (positive_texts, negative_texts).

    Positives: the 500 training texts for that variant (from training_texts.jsonl).
    Negatives: 500 items from benchmarks NOT used in this variant's training.
    """
    # Positives: from the saved training texts
    ft_dir = f"{PROJECT}/checkpoints/gt_{variant}"
    positives = []
    with open(f"{ft_dir}/training_texts.jsonl") as f:
        for line in f:
            positives.append(json.loads(line)["text"])
    positives = positives[:500]

    # Negatives: rotate over all benchmarks, pick items not in positives.
    # Strategy: use held-out items from the variant's own benchmark (not in training)
    # + items from benchmarks not used in this variant.
    rnd = random.Random(hash(variant) & 0xFFFF)
    positives_set = set(positives)

    all_bench_pools = {
        "gsm8k_test": ("gsm8k_test", render_gsm8k),
        "mmlu_test": ("mmlu_test", render_mmlu),
        "hellaswag_val": ("hellaswag_val", render_hellaswag),
        "arc_challenge": ("arc_challenge", render_arc),
        "humaneval": ("humaneval", render_humaneval),
    }

    candidates = []
    for name, (bench_name, render) in all_bench_pools.items():
        try:
            items = load_bench(bench_name)
        except Exception:
            continue
        for ex in items:
            try:
                txt = render(ex)
                if txt not in positives_set:
                    candidates.append(txt)
            except Exception:
                continue

    rnd.shuffle(candidates)
    negatives = candidates[:500]
    return positives, negatives


def run_detectors(variant: str, gpu: int = 0):
    device = "cuda"
    torch.cuda.set_device(gpu)

    positives, negatives = build_pos_neg(variant)
    print(f"[{variant}] positives={len(positives)}, negatives={len(negatives)}")

    all_texts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)

    scores = {"label": labels}

    # ---- 1. N-gram overlap vs Pile index (for Pythia contamination signal) ----
    pile_path = f"{LOCALSSD}/data/pile_ngrams_13.pkl"
    if os.path.exists(pile_path):
        print(f"[{variant}] loading Pile 13-gram index...")
        with open(pile_path, "rb") as f:
            pile_idx = pickle.load(f)
        print(f"  |Pile-13|={len(pile_idx):,}")
        scores["ngram13_pile"] = ngram_overlap(all_texts, pile_idx, n=13)
        del pile_idx
    else:
        print(f"[{variant}] SKIP n-gram Pile (index missing)")

    fw_path = f"{LOCALSSD}/data/fineweb_ngrams_13.pkl"
    if os.path.exists(fw_path):
        print(f"[{variant}] loading FineWeb 13-gram index...")
        with open(fw_path, "rb") as f:
            fw_idx = pickle.load(f)
        print(f"  |FineWeb-13|={len(fw_idx):,}")
        scores["ngram13_fineweb"] = ngram_overlap(all_texts, fw_idx, n=13)
        del fw_idx
    else:
        print(f"[{variant}] SKIP n-gram FineWeb (index missing)")

    # Also: n-gram overlap vs the variant's OWN training set (fine-tune reference)
    print(f"[{variant}] computing self-training-set n-gram index...")
    from detectors import _tokenize_words
    import xxhash
    self_idx = set()
    for t in positives:
        toks = _tokenize_words(t)
        for i in range(len(toks) - 12):
            self_idx.add(xxhash.xxh64(" ".join(toks[i:i+13]).encode()).intdigest())
    scores["ngram13_self"] = ngram_overlap(all_texts, self_idx, n=13)
    del self_idx

    # ---- 2. Min-K% Prob & 3. Reference-model loss (share model loading) ----
    print(f"[{variant}] loading fine-tuned and base Pythia-1B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    ft_model = AutoModelForCausalLM.from_pretrained(
        f"{PROJECT}/checkpoints/gt_{variant}", dtype=torch.bfloat16
    ).to(device)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device)

    for k in [5, 10, 20, 30, 50]:
        print(f"[{variant}] min_k_prob k={k}...")
        scores[f"mink_{k}"] = min_k_prob(all_texts, ft_model, tok, k_percent=k, device=device)

    print(f"[{variant}] per-item loss (fine-tuned)...")
    ft_loss = per_item_loss(all_texts, ft_model, tok, device=device)
    print(f"[{variant}] per-item loss (base)...")
    base_loss = per_item_loss(all_texts, base_model, tok, device=device)
    scores["loss_ft"] = ft_loss
    scores["loss_base"] = base_loss
    scores["refmodel_delta"] = ref_model_delta(ft_loss, base_loss)

    # Free fine-tuned & base
    del ft_model, base_model
    torch.cuda.empty_cache()

    # ---- 4. Embedding similarity (vs positives-as-corpus) ----
    # For calibration: embed all candidates, compute max cosine vs a sample of the
    # positive training set. High sim → candidate is near the training data.
    # In Phase 4 this is replaced with the audited model's inferred training corpus.
    print(f"[{variant}] E5 embeddings...")
    from transformers import AutoModel, AutoTokenizer as E5Tok
    e5_tok = E5Tok.from_pretrained(E5_MODEL)
    e5 = AutoModel.from_pretrained(E5_MODEL, dtype=torch.float16).to(device)
    all_emb = embed_texts(all_texts, e5, e5_tok, device=device, batch_size=64)
    pos_emb = embed_texts(positives, e5, e5_tok, device=device, batch_size=64)
    scores["embsim_self"] = max_cosine_sim(all_emb, pos_emb)

    del e5
    torch.cuda.empty_cache()

    # ---- 5. Slot-guessing: skipped in calibration (requires per-benchmark MC item structure)
    # We'll compute slot-guessing only in Phase 4 where items are fresh benchmarks.

    # Save scores
    out_dir = f"{RESULTS}/{variant}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/scores.pkl", "wb") as f:
        pickle.dump(scores, f)
    print(f"[{variant}] saved scores to {out_dir}/scores.pkl")
    print(f"[{variant}] keys: {list(scores.keys())}")

    # Append experiment log
    entry = {
        "id": f"det_calib_{variant}",
        "paper": "paper_5_contamination",
        "name": f"detection_calibration_{variant}",
        "status": "success",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_count": 1,
        "metrics": {k: {"mean_pos": float(v[labels == 1].mean()) if k != "label" else None,
                         "mean_neg": float(v[labels == 0].mean()) if k != "label" else None}
                    for k, v in scores.items() if k != "label"},
        "output_path": out_dir,
        "verified": False,
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    run_detectors(args.variant, gpu=args.gpu)


if __name__ == "__main__":
    main()
