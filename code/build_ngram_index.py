#!/usr/bin/env python3
"""Sample documents from a training corpus (Pile or Dolma) via HF streaming,
generate 13-gram and 8-gram hash sets, save to disk for fast n-gram overlap queries.

Usage:
    python build_ngram_index.py --corpus pile --n_docs 500000
    python build_ngram_index.py --corpus dolma --n_docs 500000

Output: /mnt/localssd/data/{corpus}_ngrams_13.pkl and _ngrams_8.pkl
        Each file is a pickled set[int] of xxhash64 digests.
"""
import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import xxhash
from datasets import load_dataset

OUT_DIR = "/mnt/localssd/data"
os.makedirs(OUT_DIR, exist_ok=True)

WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str):
    """Whitespace-ish word tokenization for n-gram computation.
    Consistent with standard contamination detection literature
    (word-level n-grams, lowercased, non-alpha stripped)."""
    return [t.lower() for t in WORD_RE.findall(text)]


def ngram_hashes(tokens, n):
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i:i + n])
        yield xxhash.xxh64(ngram.encode("utf-8")).intdigest()


def stream_corpus(corpus):
    if corpus == "pile":
        # Pile-uncopyrighted: deduped Pile without copyrighted subsets. Standard reference for Pythia.
        for ex in load_dataset("monology/pile-uncopyrighted", split="train", streaming=True):
            yield ex.get("text") or ""
    elif corpus == "fineweb":
        # Dolma proxy: FineWeb (CC-based, similar register to Dolma). allenai/dolma is closed.
        for ex in load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True):
            yield ex.get("text") or ""
    elif corpus == "c4":
        for ex in load_dataset("allenai/c4", "en", split="train", streaming=True):
            yield ex.get("text") or ""
    else:
        raise ValueError(f"Unknown corpus {corpus}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["pile", "fineweb", "c4"])
    ap.add_argument("--n_docs", type=int, default=500_000)
    ap.add_argument("--max_tokens_per_doc", type=int, default=2000,
                    help="Cap tokens per document to bound memory/time")
    ap.add_argument("--n_values", nargs="+", type=int, default=[13, 8])
    args = ap.parse_args()

    print(f"[{args.corpus}] starting; n_docs={args.n_docs}, ngrams={args.n_values}")
    sets = {n: set() for n in args.n_values}
    t0 = time.time()
    n_seen = 0
    n_tokens = 0
    try:
        for text in stream_corpus(args.corpus):
            tokens = tokenize(text)[:args.max_tokens_per_doc]
            if len(tokens) < max(args.n_values):
                continue
            n_seen += 1
            n_tokens += len(tokens)
            for n in args.n_values:
                sets[n].update(ngram_hashes(tokens, n))
            if n_seen % 5000 == 0:
                dt = time.time() - t0
                sizes = {n: len(s) for n, s in sets.items()}
                print(f"  {n_seen:>7} docs, {n_tokens/1e6:.1f}M tokens, {dt:.0f}s, sizes={sizes}")
            if n_seen >= args.n_docs:
                break
    except Exception as e:
        print(f"Streaming interrupted after {n_seen} docs: {type(e).__name__}: {e}")

    # Save
    for n in args.n_values:
        out = f"{OUT_DIR}/{args.corpus}_ngrams_{n}.pkl"
        with open(out, "wb") as f:
            pickle.dump(sets[n], f, protocol=4)
        size_mb = os.path.getsize(out) / 1e6
        print(f"[SAVED] {out}: |set|={len(sets[n]):,}  file={size_mb:.0f} MB")

    # Also save a manifest
    manifest = {
        "corpus": args.corpus,
        "n_docs_processed": n_seen,
        "n_tokens": n_tokens,
        "ngram_sizes": args.n_values,
        "unique_ngrams": {n: len(sets[n]) for n in args.n_values},
        "duration_sec": time.time() - t0,
    }
    with open(f"{OUT_DIR}/{args.corpus}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
