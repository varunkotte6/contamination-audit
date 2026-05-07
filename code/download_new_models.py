#!/usr/bin/env python3
"""Download 4 additional audit models to expand the 14-model matrix.

Targets: Qwen2.5-32B, Llama-3.3-70B-Instruct (for 4-bit quant at eval),
Qwen3-8B, Phi-3.5-mini-instruct. If a model is gated or unavailable,
we skip and log the failure.
"""
import os
import sys
import time
import traceback

MODELS = [
    ("Qwen/Qwen2.5-32B", "Qwen__Qwen2.5-32B"),
    ("meta-llama/Llama-3.3-70B-Instruct", "meta-llama__Llama-3.3-70B-Instruct"),
    ("Qwen/Qwen3-8B", "Qwen__Qwen3-8B"),
    ("microsoft/Phi-3.5-mini-instruct", "microsoft__Phi-3.5-mini-instruct"),
]

LOCALSSD = "/mnt/localssd/models"
os.makedirs(LOCALSSD, exist_ok=True)


def download(mid, short):
    from huggingface_hub import snapshot_download
    path = f"{LOCALSSD}/{short}"
    if os.path.exists(path) and os.listdir(path):
        print(f"[{mid}] already present at {path}; skipping")
        return True
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=mid,
            local_dir=path,
            token=os.environ.get("HF_TOKEN"),
            max_workers=8,
        )
        print(f"[{mid}] downloaded in {time.time()-t0:.0f}s")
        return True
    except Exception as e:
        print(f"[{mid}] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    results = {}
    for mid, short in MODELS:
        print(f"\n=== {mid} ===")
        results[mid] = download(mid, short)
    print("\n=== summary ===")
    for mid, ok in results.items():
        print(f"  {mid}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
