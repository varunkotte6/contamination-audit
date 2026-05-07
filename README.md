# The Calibration Mismatch in LLM Contamination Detection

Reproducible pipeline + released per-example contamination database for
our NeurIPS 2026 D&B submission:
**"The Calibration Mismatch in LLM Contamination Detection: Why Prior Methods Invert, and a Simple Fix"**.

## Headline

Per-example contamination detectors achieve $F_1 \approx 1$ on standard
calibration setups, yet the \emph{same} calibrated detectors produce
\emph{inverted} accuracy signals at deployment on 12 open-weight LLMs
across uniform-difficulty benchmarks (v1: mean clean − contam gap
$+9.1$ pp wrong direction, $t$-test $p=0.001$). We trace the failure
to a domain-vs-memorization confound introduced by drawing calibration
negatives from a different benchmark than the positives. Switching
to same-benchmark held-out negatives (v2) preserves calibration
$F_1 = 0.9995$ and eliminates the inversion bias
(Mann-Whitney $p=0.001$ that v1 > v2 on uniform-difficulty benchmarks).

## Quick links

- `paper/main.tex` — 7-section paper draft
- `paper/refs.bib` — bibliography (18 citations)
- `paper/datasheet.md` — D&B datasheet for released dataset
- `figures_v2/` — publication-grade figures + tables (PDF + CSV + TeX)
- `code/` — full pipeline (fine-tune, detect, calibrate, audit, eval, figures)
- `results/` — per-model, per-benchmark contamination scores + accuracy

## Released artifacts

| Artifact | Path | Notes |
|---|---|---|
| Per-example contamination scores (v1 ensemble) | `results/phase4/` | 12 models × 13 benchmarks, ~113K cells |
| Per-example contamination scores (v2 ensemble) | `results/phase4_v2/` | Same matrix, v2 calibration |
| Trained v2 ensemble | `results/phase3_v2/ensemble.pkl` | Scikit-learn scaler + LR + sign metadata |
| Ground-truth variants | `checkpoints/gt_{A,B,C,D,E}/` | Pythia-1B fine-tunes (50-epoch + light-regime 1/5/10-ep) |
| Calibration scores | `results/phase3_v2/{A,B,C,D,E}/scores.pkl` | 12-feature score matrices |

## Reproducing the pipeline

Prerequisites: 2× H100 or equivalent; `conda`, `python 3.11`, HuggingFace
token with Meta-Llama-3.1 and Qwen-2.5 license acceptance (Meta-Llama-3-8B
is NOT required; was not audited).

```bash
# 1. Environment
conda create -p envs/paper5 python=3.11 -y
conda activate envs/paper5
pip install torch transformers datasets accelerate peft trl \
           vllm lm-eval bitsandbytes xxhash \
           matplotlib seaborn scipy scikit-learn pandas numpy \
           jsonlines wandb huggingface_hub

# 2. Download models + benchmarks
HF_TOKEN=hf_xxx bash setup.sh 5         # project-specific setup

# 3. Build n-gram reference indices (Pile + FineWeb 500K docs each)
python code/build_ngram_index.py --corpus pile    --n_docs 500000
python code/build_ngram_index.py --corpus fineweb --n_docs 500000

# 4. Ground-truth fine-tunes (5 variants, 2-3h on 2×H100)
bash code/run_phase2.sh

# 5. Detector calibration (v2 is the default fix)
bash code/run_phase3_v2.sh
# Also optional: bash code/run_lightreg_detect.sh for exposure sweep

# 6. Full audit (12 models × 13 benchmarks)
bash code/run_phase4.sh
python code/phase4_reapply_v2.py   # rewrites p_contam with v2 ensemble

# 7. Per-item accuracy evaluation
bash code/run_phase5_full.sh        # GSM8K, MMLU, HellaSwag, ARC, HumanEval (prefix match)
bash code/run_sandbox_humaneval.sh  # HumanEval pass@1 with subprocess exec
bash code/run_math_eval.sh          # MATH-500 numeric-boxed
bash code/run_lcb_sandbox.sh        # LiveCodeBench pass@1 (subprocess + stdin/stdout)

# 8. Analysis + figures + tables
python code/analyze_inversion.py
python code/stat_tests_v2.py
python code/polish_figures.py
```

## Scope + caveats

- **Models audited (12):** Pythia-{70M, 410M, 1B, 2.8B, 6.9B}, OLMo-2-7B,
  Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B-v0.3, Gemma-2-9B, Phi-4,
  DeepSeek-R1-Distill-Qwen-7B. DeepSeek-V2-Lite dropped
  (transformers-5.x incompatibility in its custom modeling code).
  Meta-Llama-3-8B dropped (HF token license).
- **Benchmarks (13):** GSM8K, MATH, MATH-500, MMLU, HumanEval, MBPP,
  ARC-Challenge, HellaSwag, BBH, AIME-2024, TruthfulQA, AGIEval,
  LiveCodeBench. GPQA dropped (gated; no open mirror). SWE-bench-lite
  dropped (agentic eval is separate infrastructure).
- **Corpora for n-gram references:** The Pile (for Pythia) and FineWeb
  (proxy for Dolma-trained OLMo-2 since the Dolma HF loader was
  deprecated during the audit window).
- **Accuracy metrics:** sandboxed pass@1 for HumanEval + LiveCodeBench
  (subprocess with 5-sec timeout against public test cases), numeric
  boxed-match for MATH, last-number for GSM8K, first-letter match for
  MC benchmarks. The prefix-match approximation in the original Phase
  5 full is superseded by the sandboxed runs where available.

## Citation

```
@inproceedings{anon2026calibration,
  title   = {The Calibration Mismatch in LLM Contamination Detection:
             Why Prior Methods Invert, and a Simple Fix},
  author  = {Anonymous},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track (under review)},
  year    = {2026},
}
```

## License

- Code: Apache 2.0
- Released scores/artifacts: CC BY 4.0
- Underlying benchmarks retain original licenses.
