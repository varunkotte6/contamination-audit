# File-format and schema documentation

This document specifies the on-disk layout of the released artifacts.
All paths are relative to the project root.

## Per-model contamination scores (`results/phase4_v2/{model_short}/scores.pkl`)

Pickle of a Python dict with keys:

| Key | Type | Shape | Description |
|---|---|---|---|
| `model_id` | str | — | HuggingFace model ID (e.g., `allenai/OLMo-2-1124-7B`) |
| `features` | dict[str, np.ndarray] | each (N,) float32 | 10 feature values per item |
| `p_contam` | np.ndarray | (N,) float32 | ensemble contamination probability per item |
| `provenance` | list[(str, int)] | N | `(benchmark_name, original_item_idx)` per row |
| `_ensemble_version` | str | — | `"v2_same_bench_negatives"` |

Where `N ≈ 9,000–10,000` (varies per model due to benchmark inclusion).

The `features` dict contains:
`ngram13_pile`, `ngram13_fineweb`, `mink_5`, `mink_10`, `mink_20`,
`mink_30`, `mink_50`, `loss_ft`, `loss_base`, `refmodel_delta`.

## Per-model summary (`results/phase4_v2/{model_short}/summary.json`)

JSON object keyed by benchmark name, each value:

```json
{
  "n_items": int,
  "p_contam_mean": float,
  "p_contam_p90": float,
  "high_risk_rate": float,
  "mean_loss_audit": float,
  "mean_refmodel_delta": float
}
```

## Calibrated ensemble (`results/phase3_v2/ensemble.pkl`)

Pickle of a dict:
```python
{
  "scaler": sklearn.preprocessing.StandardScaler,  # fitted on dev variants
  "clf":    sklearn.linear_model.LogisticRegression,
  "methods": list[str],   # 10-element feature order matching scaler/clf
  "signs":  dict[str, float],  # per-feature sign (+1/-1) after AUC auto-flip
}
```

## Per-cell statistical tests (`results/stat_tests/stat_tests_v2.csv`)

CSV with one row per (model, benchmark, ensemble) triple that passed
$n_{\text{clean}} \geq 10$ and $n_{\text{contam}} \geq 10$:

| Column | Description |
|---|---|
| `model` | HF model id |
| `benchmark` | benchmark name (with suffix `_sandbox` where applicable) |
| `ensemble` | `"v1"` or `"v2"` |
| `evaluator` | `"phase5_full"` \| `"sandbox_pass1"` \| `"numeric_boxed"` |
| `n_clean`, `n_contam` | subset sizes |
| `clean_acc`, `contam_acc` | within-subset mean accuracy |
| `obs_gap` | `clean_acc − contam_acc` (positive = wrong direction) |
| `ci_low`, `ci_high` | 95% bootstrap CI on the gap (2000 resamples) |
| `perm_p` | one-sided 5000-sample permutation p-value: `P(contam > clean)` |

## Ground-truth fine-tune checkpoints (`checkpoints/gt_{A,B,C,D,E}/`)

Each directory contains:
- `config.json`, `generation_config.json`, `model.safetensors`,
  `tokenizer.json`, `tokenizer_config.json` — standard HF snapshot
- `metrics.json` — `{"verbatim_rate": float, "duration_sec": float, ...}`
- `training_texts.jsonl` — the 500 texts memorized by this variant

## Light-regime fine-tune checkpoints (`checkpoints/gt_{A,B,C,D}_{1,5,10}ep/`)

Same structure as above, one directory per (variant, epoch-count) pair.

## Random seeds

- Phase 2 fine-tune: framework default (Hugging Face Trainer, seed=42
  via `TrainingArguments.seed`).
- Phase 3 pos/neg sampling: `random.Random(hash(("v2", variant)) & 0xFFFF)`
  per variant.
- N-gram index: no seed (deterministic pipeline).
- Bootstrap / permutation tests: `np.random.default_rng(0)`.

## File formats

| Extension | Library |
|---|---|
| `.jsonl` | one JSON object per line (utf-8) |
| `.pkl` | Python pickle, protocol 4 (binary) |
| `.safetensors` | HuggingFace safetensors format (bf16 weights) |
| `.csv` | UTF-8 with header row |
| `.npz` | NumPy binary archive |

## Loading example

```python
import pickle
import numpy as np

# Load per-model contamination scores
with open("results/phase4_v2/microsoft__phi-4/scores.pkl", "rb") as f:
    d = pickle.load(f)

p_contam = d["p_contam"]                          # np.ndarray (N,)
benchmark_of_item = [p[0] for p in d["provenance"]]

# Select GSM8K items
gsm_mask = np.array([b == "gsm8k_test" for b in benchmark_of_item])
print(f"{gsm_mask.sum()} GSM8K items; mean p_contam = {p_contam[gsm_mask].mean():.3f}")

# Apply the ensemble to new features manually
with open("results/phase3_v2/ensemble.pkl", "rb") as f:
    ens = pickle.load(f)
# new_features: np.ndarray (M, 10) in the order of ens["methods"]
# p_new = ens["clf"].predict_proba(ens["scaler"].transform(new_features))[:, 1]
```
