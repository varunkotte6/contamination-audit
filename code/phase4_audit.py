#!/usr/bin/env python3
"""Phase 4 — Full contamination audit.

For a given audit model, compute per-item contamination features on every
benchmark item and apply the calibrated ensemble to produce a contamination
probability per item.

Features computed per (model, item):
  - mink_{5,10,20,30,50}:    Min-K% Prob on the audited model
  - loss_audit:              per-item CE loss under the audit model
  - loss_ref (Pythia-1B):    per-item CE loss under the Pythia-1B reference
  - refmodel_delta:          loss_ref - loss_audit
  - ngram13_pile:            13-gram overlap with Pile (for models trained on Pile)
  - ngram13_fineweb:         13-gram overlap with FineWeb (web-corpus proxy)
  - ngram13_self:            placeholder (0 during audit; meaningful only for GT calibration)
  - embsim_self:             NaN (no self-corpus for audit models)

Usage:
    python phase4_audit.py --model_id EleutherAI/pythia-1b \\
                            --model_path /mnt/localssd/models/EleutherAI__pythia-1b \\
                            --gpu 0
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from detectors import (
    ngram_overlap, render_benchmark_item,
    batched_token_logprobs, min_k_prob_from_logprobs, per_item_loss_from_logprobs,
    ref_model_delta,
)

PROJECT = "{REPO_ROOT}"
LOCALSSD = "/mnt/localssd"
BENCH_DIR = f"{LOCALSSD}/data/benchmarks"
REF_MODEL = f"{LOCALSSD}/models/EleutherAI__pythia-1b"
RESULTS = f"{PROJECT}/results/phase4"
Path(RESULTS).mkdir(parents=True, exist_ok=True)

BENCHMARKS = [
    "gsm8k_test", "math", "math500", "mmlu_test", "humaneval", "mbpp",
    "arc_challenge", "hellaswag_val", "bbh", "aime2024", "truthfulqa_mc",
    "agieval", "livecodebench",
]


def load_bench(name):
    path = f"{BENCH_DIR}/{name}.jsonl"
    if not os.path.exists(path):
        return []
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def compute_ngram_features(texts, n=13):
    feats = {}
    for name, path in [
        ("ngram13_pile", f"{LOCALSSD}/data/pile_ngrams_{n}.pkl"),
        ("ngram13_fineweb", f"{LOCALSSD}/data/fineweb_ngrams_{n}.pkl"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                idx = pickle.load(f)
            feats[name] = ngram_overlap(texts, idx, n=n)
            del idx
        else:
            feats[name] = np.full(len(texts), np.nan, dtype=np.float32)
    return feats


def run_audit(model_id: str, model_path: str):
    device = "cuda"
    short_name = model_id.replace("/", "__")
    out_dir = f"{RESULTS}/{short_name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== AUDIT {model_id} ===")
    t_model_start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        audit_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
    except Exception as e:
        print(f"[FAIL LOAD] {model_id}: {type(e).__name__}: {e}")
        return

    # Render all benchmark items to text, tagged with benchmark + item index
    all_texts = []
    provenance = []  # list of (benchmark, idx)
    for bench in BENCHMARKS:
        items = load_bench(bench)
        for i, ex in enumerate(items):
            try:
                txt = render_benchmark_item(bench, ex)
                all_texts.append(txt)
                provenance.append((bench, i))
            except Exception:
                continue
    print(f"[{model_id}] {len(all_texts)} items across {len(set(p[0] for p in provenance))} benchmarks")

    # ---- Single batched forward pass → per-item loss + all Min-K% values ----
    # Batch size heuristic: smaller for larger models (memory-bound).
    n_params = sum(p.numel() for p in audit_model.parameters())
    if n_params > 20e9:    # >20B → MoE / 14B dense
        batch_size = 2
    elif n_params > 7e9:    # 7-9B
        batch_size = 4
    elif n_params > 2e9:    # 2-7B
        batch_size = 8
    else:                     # <2B
        batch_size = 16
    print(f"[{model_id}] audit forward pass (bs={batch_size}, {n_params/1e9:.1f}B params)...")
    t0 = time.time()
    audit_lps = batched_token_logprobs(all_texts, audit_model, tok, batch_size=batch_size,
                                         device=device)
    dt_audit = time.time() - t0
    print(f"  done in {dt_audit:.0f}s")

    loss_audit = per_item_loss_from_logprobs(audit_lps)
    mink = {f"mink_{k}": min_k_prob_from_logprobs(audit_lps, k_percent=k)
            for k in [5, 10, 20, 30, 50]}
    del audit_lps, audit_model
    torch.cuda.empty_cache()

    # ---- Reference Pythia-1B loss (cached to disk since it's identical across audits) ----
    ref_cache = f"{PROJECT}/results/phase4/_ref_loss_pythia1b.npz"
    ref_provenance_cache = f"{PROJECT}/results/phase4/_ref_provenance.json"
    loss_ref = None
    if os.path.exists(ref_cache) and os.path.exists(ref_provenance_cache):
        with open(ref_provenance_cache) as f:
            cached_prov = json.load(f)
        if cached_prov == [list(p) for p in provenance]:
            print(f"[{model_id}] using cached reference loss")
            loss_ref = np.load(ref_cache)["loss_ref"]
    if loss_ref is None:
        print(f"[{model_id}] computing reference Pythia-1B loss...")
        ref_tok = AutoTokenizer.from_pretrained(REF_MODEL)
        ref_tok.pad_token = ref_tok.eos_token
        ref_model = AutoModelForCausalLM.from_pretrained(REF_MODEL, dtype=torch.bfloat16).to(device)
        t0 = time.time()
        ref_lps = batched_token_logprobs(all_texts, ref_model, ref_tok, batch_size=16, device=device)
        loss_ref = per_item_loss_from_logprobs(ref_lps)
        print(f"  done in {time.time()-t0:.0f}s")
        np.savez(ref_cache, loss_ref=loss_ref)
        with open(ref_provenance_cache, "w") as f:
            json.dump([list(p) for p in provenance], f)
        del ref_model, ref_lps
        torch.cuda.empty_cache()

    refmodel_delta = loss_ref - loss_audit

    # ---- N-gram overlap features ----
    print(f"[{model_id}] n-gram overlap features...")
    t0 = time.time()
    ngram_feats = compute_ngram_features(all_texts, n=13)
    print(f"  done in {time.time()-t0:.0f}s")

    # Assemble per-item feature matrix. Feature names MUST match those used
    # during calibration (phase3_detect.py) so the ensemble can find them.
    features = {
        "loss_ft": loss_audit,       # calibration name was loss_ft (fine-tuned model loss)
        "loss_base": loss_ref,        # calibration name was loss_base
        "refmodel_delta": refmodel_delta,
        **mink,
        **ngram_feats,
    }

    # Apply calibrated ensemble (audit-deployable variant — only features present
    # in both calibration and audit settings).
    ens_path = f"{PROJECT}/results/phase3/ensemble.pkl"
    p_contam = np.full(len(all_texts), np.nan, dtype=np.float32)
    if os.path.exists(ens_path):
        with open(ens_path, "rb") as f:
            ens = pickle.load(f)
        methods = ens["methods"]
        # Note: ensemble was trained on RAW (unflipped) scores; do NOT apply signs
        # when assembling the feature matrix here. Signs are only used for per-method
        # thresholded baselines, not for the logistic-regression ensemble.
        cols = []
        missing = []
        for m in methods:
            if m in features:
                cols.append(features[m])
            else:
                missing.append(m)
                cols.append(np.full(len(all_texts), np.nan, dtype=np.float32))
        if missing:
            print(f"[{model_id}] WARN: missing features in audit: {missing}")
        X = np.stack(cols, axis=1)
        # Impute NaN per-column with the column mean from dev (scaler.mean_)
        for j in range(X.shape[1]):
            col = X[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col[nan_mask] = ens["scaler"].mean_[j]
        X_s = ens["scaler"].transform(X)
        p_contam = ens["clf"].predict_proba(X_s)[:, 1]

    # Save per-benchmark contamination scores
    save_data = {
        "model_id": model_id,
        "features": {k: v.astype(np.float32) for k, v in features.items()},
        "p_contam": p_contam.astype(np.float32),
        "provenance": provenance,  # list of (benchmark, item_idx)
    }
    with open(f"{out_dir}/scores.pkl", "wb") as f:
        pickle.dump(save_data, f)

    # Summary stats per benchmark
    summary = {}
    prov_arr = np.array(provenance, dtype=object)
    benchmarks_present = sorted(set(p[0] for p in provenance))
    for b in benchmarks_present:
        mask = np.array([p[0] == b for p in provenance])
        summary[b] = {
            "n_items": int(mask.sum()),
            "p_contam_mean": float(np.nanmean(p_contam[mask])) if not np.all(np.isnan(p_contam[mask])) else None,
            "p_contam_p90": float(np.nanpercentile(p_contam[mask], 90)) if not np.all(np.isnan(p_contam[mask])) else None,
            "high_risk_rate": float((p_contam[mask] > 0.5).mean()) if not np.all(np.isnan(p_contam[mask])) else None,
            "mean_loss_audit": float(np.nanmean(loss_audit[mask])),
            "mean_refmodel_delta": float(np.nanmean(refmodel_delta[mask])),
        }
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    dt = time.time() - t_model_start
    print(f"=== DONE {model_id} in {dt/60:.1f} min ===")
    for b, s in summary.items():
        print(f"  {b}: n={s['n_items']}, p_contam_mean={s['p_contam_mean']}, high_risk_rate={s['high_risk_rate']}")

    # Append experiment log
    entry = {
        "id": f"audit_{short_name}",
        "paper": "paper_5_contamination",
        "name": f"audit_{short_name}",
        "status": "success",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": dt,
        "gpu_hours": dt / 3600,
        "output_path": out_dir,
        "verified": False,
        "summary": summary,
    }
    with open(f"{PROJECT}/experiments/experiment_log.jsonl", "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="HF model ID (used for dir naming)")
    ap.add_argument("--model_path", required=True, help="Local path to model weights")
    args = ap.parse_args()
    run_audit(args.model_id, args.model_path)


if __name__ == "__main__":
    main()
