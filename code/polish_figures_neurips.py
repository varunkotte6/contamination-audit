#!/usr/bin/env python3
"""Generate clean publication-quality PDF figures for the NeurIPS 2026 paper.

Style: serif font (Computer Modern-like via STIX), 9pt base, 6x3 inch default
figure size fits NeurIPS text width of 5.5 inches with modest margin.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROJECT = "{REPO_ROOT}"
OUT = f"{PROJECT}/paper/neurips2026/figures"
Path(OUT).mkdir(parents=True, exist_ok=True)

# Consistent publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.alpha": 0.25,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = {
    "v1": "#d62728",   # red
    "v2": "#2ca02c",   # green
    "neutral": "#7f7f7f",
    "llama": "#1f77b4",
    "qwen": "#ff7f0e",
    "olmo": "#9467bd",
    "mistral": "#8c564b",
    "gemma": "#e377c2",
    "phi": "#17becf",
    "pythia": "#bcbd22",
}


def fig1_calibration_f1():
    """F1 bars for v1 vs v2 across calibration variants."""
    variants = ["A (GSM8K)", "B (MMLU)", "C (HellaSwag)", "D (mixed)", "E (Wiki)"]
    f1_v1 = [0.9995, 0.9995, 0.9995, 0.9995, 0.9975]
    f1_v2 = [1.000, 1.000, 1.000, 0.999, 0.998]
    x = np.arange(len(variants))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    ax.bar(x - w/2, f1_v1, width=w, color=PALETTE["v1"], label="v1 (cross-benchmark)")
    ax.bar(x + w/2, f1_v2, width=w, color=PALETTE["v2"], label="v2 (same-benchmark)")
    ax.set_ylim(0.99, 1.001)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=0)
    ax.set_ylabel("Calibration $F_1$")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Both calibrations saturate at $F_1\\approx 1$")
    fig.savefig(f"{OUT}/fig1_calibration_f1.pdf")
    plt.close(fig)
    print(f"Wrote {OUT}/fig1_calibration_f1.pdf")


def fig2_deployment_inversion():
    """Distribution of per-cell gaps for v1 vs v2 on uniform-difficulty.

    Filter matches Table 1: 14-model core panel, uniform-difficulty
    capability benchmarks (TruthfulQA excluded), n>=10 in both subsets.
    """
    try:
        df = pd.read_csv(f"{PROJECT}/results/stat_tests/stat_tests_v2.csv")
    except FileNotFoundError:
        print("stat_tests_v2.csv missing; skip fig2")
        return
    UNIFORM = {"gsm8k_test", "mmlu_test", "hellaswag_val", "arc_challenge",
               "humaneval", "humaneval_sandbox", "mbpp_sandbox"}
    CORE14 = {"EleutherAI/pythia-70m", "EleutherAI/pythia-410m",
              "EleutherAI/pythia-1b", "EleutherAI/pythia-2.8b",
              "EleutherAI/pythia-6.9b", "allenai/OLMo-2-1124-7B",
              "meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.3",
              "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "google/gemma-2-9b",
              "google/gemma-2-27b", "microsoft/phi-4",
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
    df = df[df.benchmark.isin(UNIFORM) & df.model.isin(CORE14)
             & (df.n_clean >= 10) & (df.n_contam >= 10)]
    v1 = df[df.ensemble == "v1"].obs_gap.values
    v2 = df[df.ensemble == "v2"].obs_gap.values
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    bins = np.linspace(-0.7, 0.7, 28)
    ax.hist(v1, bins=bins, alpha=0.6, color=PALETTE["v1"],
            label=f"v1 (n={len(v1)}), mean={v1.mean():+.3f}", density=True)
    ax.hist(v2, bins=bins, alpha=0.6, color=PALETTE["v2"],
            label=f"v2 (n={len(v2)}), mean={v2.mean():+.3f}", density=True)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Per-cell accuracy gap (clean minus contam), proportion")
    ax.set_ylabel("Density")
    ax.set_title("v1 inverts (positive gap, wrong direction); v2 centers below zero (correct)")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(f"{OUT}/fig2_gap_distribution.pdf")
    plt.close(fig)
    print(f"Wrote {OUT}/fig2_gap_distribution.pdf")


def fig3_bonferroni_core():
    """Bar chart of Bonferroni-surviving counts across 9 specifications."""
    specs = [("Pythia-1B", "full", 5, 3),
             ("Pythia-1B", "no-refdelta", 5, 4),
             ("Pythia-1B", "pure-mem", 3, 3),
             ("Qwen-0.5B", "full", 5, 5),
             ("Qwen-0.5B", "no-refdelta", 6, 6),
             ("Qwen-0.5B", "pure-mem", 7, 6),
             ("OLMo-1B", "full", 6, 6),
             ("OLMo-1B", "no-refdelta", 6, 6),
             ("OLMo-1B", "pure-mem", 4, 4)]
    labels = [f"{b}\n{f}" for b, f, _, _ in specs]
    cells = [c for _, _, c, _ in specs]
    fams = [f for _, _, _, f in specs]
    x = np.arange(len(specs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    ax.bar(x - w/2, cells, width=w, color=PALETTE["v2"], label="Cells")
    ax.bar(x + w/2, fams, width=w, color=PALETTE["llama"], label="Families")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Count")
    ax.set_title("Bonferroni-surviving cells and families across 9 specifications")
    ax.legend(loc="upper left", frameon=False)
    ax.axhline(2, color="k", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(len(specs) - 0.5, 2.2, "Universal core = 2", fontsize=7, ha="right", alpha=0.7)
    fig.savefig(f"{OUT}/fig3_bonferroni_core.pdf")
    plt.close(fig)
    print(f"Wrote {OUT}/fig3_bonferroni_core.pdf")


def fig4_temporal_flag_rate():
    """v2 flag rate vs model release date with linear regression and 95% CI.

    Models are colored by family and identified by a side legend, which
    avoids overlapping in-axes annotations when several models share a
    release date (e.g., Pythia trio at 2023.4, Qwen-2.5 trio at 2024.8).
    """
    models = [
        ("Pythia-70M",      2023.40, 0.041, "pythia",  "o"),
        ("Pythia-1B",       2023.40, 0.088, "pythia",  "s"),
        ("Pythia-6.9B",     2023.40, 0.154, "pythia",  "^"),
        ("Llama-3.1-8B",    2024.60, 0.052, "llama",   "o"),
        ("Mistral-7B-v0.3", 2024.40, 0.091, "mistral", "o"),
        ("Qwen-2.5-7B",     2024.80, 0.198, "qwen",    "o"),
        ("Qwen-2.5-14B",    2024.80, 0.203, "qwen",    "s"),
        ("Qwen-2.5-32B",    2024.80, 0.047, "qwen",    "^"),
        ("Qwen-3-8B",       2025.20, 0.001, "qwen",    "v"),
        ("OLMo-2-7B",       2024.90, 0.098, "olmo",    "o"),
        ("Gemma-2-9B",      2024.50, 0.019, "gemma",   "o"),
        ("Gemma-2-27B",     2024.50, 0.043, "gemma",   "s"),
        ("Phi-4",           2024.83, 0.107, "phi",     "o"),
    ]
    dates = np.array([m[1] for m in models])
    rates = np.array([m[2] for m in models])
    from scipy import stats as sstats
    slope, intercept, r, p_val, se = sstats.linregress(dates, rates)
    x_fit = np.linspace(dates.min() - 0.05, dates.max() + 0.05, 50)
    y_fit = intercept + slope * x_fit
    n = len(dates); df = n - 2
    s_resid = np.sqrt(np.sum((rates - (intercept + slope * dates)) ** 2) / df)
    x_mean = dates.mean()
    s_xx = np.sum((dates - x_mean) ** 2)
    se_fit = s_resid * np.sqrt(1.0 / n + (x_fit - x_mean) ** 2 / s_xx)
    t_crit = sstats.t.ppf(0.975, df)
    ci_lo = y_fit - t_crit * se_fit
    ci_hi = y_fit + t_crit * se_fit

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(x_fit, y_fit, color=PALETTE["neutral"], lw=1, zorder=2)
    ax.fill_between(x_fit, ci_lo, ci_hi, color=PALETTE["neutral"],
                     alpha=0.15, zorder=1)
    # Plot each point individually so the legend can name the model
    for name, d, r_, fam, mk in models:
        ax.scatter(d, r_, s=36, marker=mk, color=PALETTE[fam],
                    label=name, edgecolors="white", linewidths=0.6,
                    zorder=3)
    ax.set_xlabel("Approximate release date (year)")
    ax.set_ylabel("v2 flag rate ($p_{\\mathrm{contam}} > 0.5$)")
    ax.set_title(f"v2 flag rate vs. release date  "
                  f"(slope $={slope:+.3f}$/yr, $p={p_val:.2f}$, "
                  f"$n=13$; exploratory)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_ylim(-0.02, 0.24)
    # Side legend in two columns to keep figure compact
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
               frameon=False, fontsize=7, ncol=1, handletextpad=0.3,
               labelspacing=0.25)
    fig.savefig(f"{OUT}/fig4_temporal.pdf")
    plt.close(fig)
    print(f"Wrote {OUT}/fig4_temporal.pdf  "
          f"slope={slope:+.4f}/yr, p={p_val:.3f}, R²={r**2:.3f}")


def fig5_light_regime():
    """Verbatim memorization curve for Qwen-0.5B at 1/3/10/20 epochs."""
    epochs = [1, 3, 10, 20]
    verb = [0.0, 0.09, 0.75, 0.83]
    pythia_verb = [0.0, 0.62, 0.80, 0.90]
    fig, ax = plt.subplots(figsize=(4.5, 2.3))
    ax.plot(epochs, verb, "o-", color=PALETTE["qwen"], label="Qwen-2.5-0.5B")
    ax.plot(epochs, pythia_verb, "s-", color=PALETTE["pythia"], label="Pythia-1B")
    ax.axhline(0.80, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(20, 0.81, "80% target", fontsize=7, ha="right", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 20])
    ax.set_xticklabels([1, 3, 10, 20])
    ax.set_xlabel("Fine-tune epochs (log)")
    ax.set_ylabel("Verbatim reproduction")
    ax.set_title("Memorization curve by ground-truth base")
    ax.legend(loc="lower right", frameon=False)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.savefig(f"{OUT}/fig5_lightregime.pdf")
    plt.close(fig)
    print(f"Wrote {OUT}/fig5_lightregime.pdf")


def main():
    fig1_calibration_f1()
    fig2_deployment_inversion()
    fig3_bonferroni_core()
    fig4_temporal_flag_rate()
    fig5_light_regime()
    print("All figures written to:", OUT)


if __name__ == "__main__":
    main()
