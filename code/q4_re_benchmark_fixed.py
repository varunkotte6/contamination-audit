#!/usr/bin/env python3
"""Q4: model-level RE with benchmark as a fixed effect.

Reviewer asks for a specification that lets benchmarks soak up
between-benchmark variance explicitly while keeping model as the
random-effects level. This operationalizes the "benchmark-level
phenomenon" interpretation of the crossed-RE null.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT = "{REPO_ROOT}"
df = pd.read_csv(f"{PROJECT}/results/stat_tests/stat_tests_v2.csv")
UNIFORM = ["gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge",
            "humaneval", "humaneval_sandbox", "mbpp_sandbox"]
df = df[df.benchmark.isin(UNIFORM)]
df = df[(df.n_clean >= 10) & (df.n_contam >= 10)].copy()
df["v1"] = (df.ensemble == "v1").astype(int)
print(f"n cells: {len(df)}, n models: {df.model.nunique()}, "
      f"n benchmarks: {df.benchmark.nunique()}")

# Model: gap ~ v1 + benchmark_fixed + (1|model)
# v1 indicator captures the contrast; benchmark fixed effect absorbs
# between-benchmark variance; model RE captures within-model
# correlation across benchmarks.
m = smf.mixedlm("obs_gap ~ v1 + C(benchmark)", df, groups=df["model"]).fit()
print()
print("Model: obs_gap ~ v1 + C(benchmark) + (1|model)")
print(m.summary())

# Just the v1 coefficient row
v1_row = m.summary().tables[1].loc["v1"]
print(f"\nv1 coefficient: {v1_row['Coef.']}, "
      f"SE = {v1_row['Std.Err.']}, "
      f"p = {v1_row['P>|z|']}")
