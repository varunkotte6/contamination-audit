#!/usr/bin/env python3
"""Publication-quality figures for the paper.

Re-generates figures 1-5 with consistent styling (seaborn 'whitegrid',
color-blind palette, larger fonts, explicit axis labels). Saves to
figures_v2/ alongside existing rough drafts.
"""
import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

sns.set_theme(context="paper", style="whitegrid", palette="colorblind",
               font_scale=1.15, rc={"figure.dpi": 120,
                                      "savefig.dpi": 200,
                                      "axes.titlesize": 12,
                                      "axes.labelsize": 11,
                                      "xtick.labelsize": 10,
                                      "ytick.labelsize": 10,
                                      "legend.fontsize": 9,
                                      "legend.title_fontsize": 10})

PROJECT = "{REPO_ROOT}"
PHASE3 = f"{PROJECT}/results/phase3"
PHASE3_V2 = f"{PROJECT}/results/phase3_v2"
PHASE4_V1 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5_FULL = f"{PROJECT}/results/phase5_full"
FIGS = f"{PROJECT}/figures_v2"
Path(FIGS).mkdir(parents=True, exist_ok=True)


def fig1_roc():
    """Two-panel: v1 ROC (left) vs v2 ROC (right), per detection method."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, root, title in [(axes[0], PHASE3, "(a) v1 — other-benchmark negatives"),
                             (axes[1], PHASE3_V2, "(b) v2 — same-benchmark negatives")]:
        scores = {}
        for v in ["A", "B", "C"]:
            p = f"{root}/{v}/scores.pkl"
            if os.path.exists(p):
                with open(p, "rb") as f:
                    scores[v] = pickle.load(f)
        if not scores:
            continue
        methods = [k for k in next(iter(scores.values())).keys() if k != "label"]
        for m in methods:
            ss = [s[m] for s in scores.values() if m in s]
            ll = [s["label"] for s in scores.values() if m in s]
            if not ss:
                continue
            arr = np.concatenate(ss); lab = np.concatenate(ll)
            mask = ~np.isnan(arr)
            if mask.sum() == 0 or len(np.unique(lab[mask])) < 2:
                continue
            try:
                fpr, tpr, _ = roc_curve(lab[mask], arr[mask])
                # Auto-flip if AUC<0.5
                from sklearn.metrics import roc_auc_score
                if roc_auc_score(lab[mask], arr[mask]) < 0.5:
                    fpr, tpr, _ = roc_curve(lab[mask], -arr[mask])
                label = m.replace("_", " ")
                ax.plot(fpr, tpr, lw=1.2, alpha=0.7, label=label)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        if ax is axes[0]:
            ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.3)
        # Zoom to low-FPR region where the methods actually differ
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0.8, 1.005)
    axes[1].legend(loc="lower right", ncol=1, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig1_roc.pdf")
    plt.savefig(f"{FIGS}/fig1_roc.png")
    plt.close()
    print(f"[fig1] saved")


def fig2_heatmap():
    """v2 contamination heatmap, benchmarks × models."""
    rows = []
    for m in sorted(os.listdir(PHASE4_V2)):
        if m.startswith("_"):
            continue
        sp = f"{PHASE4_V2}/{m}/scores.pkl"
        if not os.path.exists(sp):
            continue
        with open(sp, "rb") as f:
            d = pickle.load(f)
        p = np.asarray(d["p_contam"])
        for (b, _), pp in zip(d["provenance"], p):
            rows.append({"model": m.replace("__", "/").replace("EleutherAI/", "").replace("allenai/", ""),
                         "benchmark": b, "p": float(pp)})
    df = pd.DataFrame(rows)
    agg = df.groupby(["benchmark", "model"]).p.mean().reset_index()
    mat = agg.pivot(index="benchmark", columns="model", values="p")

    # Order benchmarks by mean contam (descending)
    bench_order = mat.mean(axis=1).sort_values(ascending=False).index
    mat = mat.loc[bench_order]
    model_order = mat.mean(axis=0).sort_values(ascending=False).index
    mat = mat[model_order]

    plt.figure(figsize=(11, 5.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="rocket_r",
                vmin=0, vmax=mat.values.max(),
                cbar_kws={"label": r"mean $p_{\mathrm{contam}}$ (v2)"},
                annot_kws={"size": 8},
                linewidths=0.4, linecolor="white")
    plt.title(r"Per-benchmark mean contamination probability under v2 calibration")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig2_heatmap.pdf")
    plt.savefig(f"{FIGS}/fig2_heatmap.png")
    plt.close()
    print(f"[fig2] saved")


def fig3_temporal():
    """Contamination vs model release date, v1 and v2 side by side."""
    MODEL_RELEASE = {
        "EleutherAI__pythia-70m": "2023-03",
        "EleutherAI__pythia-410m": "2023-03",
        "EleutherAI__pythia-1b": "2023-03",
        "EleutherAI__pythia-2.8b": "2023-03",
        "EleutherAI__pythia-6.9b": "2023-03",
        "allenai__OLMo-2-1124-7B": "2024-11",
        "meta-llama__Llama-3.1-8B": "2024-07",
        "Qwen__Qwen2.5-7B": "2024-09",
        "mistralai__Mistral-7B-v0.3": "2024-05",
        "google__gemma-2-9b": "2024-06",
        "microsoft__phi-4": "2024-12",
        "deepseek-ai__DeepSeek-R1-Distill-Qwen-7B": "2025-01",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, root, title in [(axes[0], PHASE4_V1, "(a) v1"), (axes[1], PHASE4_V2, "(b) v2")]:
        xs, ys, labels = [], [], []
        for m, rel in MODEL_RELEASE.items():
            p = f"{root}/{m}/scores.pkl"
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                d = pickle.load(f)
            ys.append(float(np.nanmean(d["p_contam"])))
            xs.append(rel)
            labels.append(m.split("__")[-1])
        xo = sorted(set(xs))
        idx = [xo.index(x) for x in xs]
        sns.scatterplot(x=idx, y=ys, s=90, ax=ax, color=sns.color_palette("colorblind")[0])
        for i, (x, y, l) in enumerate(zip(idx, ys, labels)):
            ax.annotate(l, (x, y), fontsize=7, alpha=0.9,
                         xytext=(3, 3), textcoords="offset points")
        ax.set_xticks(range(len(xo)))
        ax.set_xticklabels(xo, rotation=30, ha="right")
        ax.set_xlabel("Model release (YYYY-MM)")
        ax.set_ylabel("mean $p_{\\mathrm{contam}}$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.03, 0.6)
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig3_temporal.pdf")
    plt.savefig(f"{FIGS}/fig3_temporal.png")
    plt.close()
    print(f"[fig3] saved")


def fig4_accuracy_inversion():
    """Key figure: clean-vs-contam accuracy gap, v1 vs v2.
    Use min_n=10 so we have enough cells for visual impact;
    stratified by uniform vs variant-difficulty for clarity."""
    df = pd.read_csv(f"{PROJECT}/results/stat_tests/stat_tests_v2.csv")
    df = df.dropna(subset=["obs_gap"])

    UNIFORM = ["gsm8k_test", "mmlu_test", "hellaswag_val",
                "arc_challenge", "humaneval", "humaneval_sandbox"]
    df["difficulty_regime"] = df.benchmark.apply(
        lambda b: "uniform" if b in UNIFORM else "variant")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    pal = {"v1": sns.color_palette("colorblind")[3],
           "v2": sns.color_palette("colorblind")[0]}
    for ax, regime in zip(axes, ["uniform", "variant"]):
        sub = df[df.difficulty_regime == regime]
        if sub.empty:
            continue
        sns.swarmplot(data=sub, x="ensemble", y="obs_gap",
                       hue="ensemble", size=8, palette=pal, dodge=False,
                       legend=False, ax=ax)
        ax.axhline(0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
        # Annotate within-group means explicitly at their own x positions
        xticks = {ens: i for i, ens in enumerate(["v1", "v2"])}
        for ens in ["v1", "v2"]:
            vals = sub[sub.ensemble == ens]["obs_gap"].values
            if len(vals) == 0:
                continue
            mean_ = vals.mean()
            ax.scatter([xticks[ens]], [mean_], marker="D",
                       color="black", s=100, zorder=10, edgecolors="white")
            ax.annotate(f"mean = {mean_:+.3f}", (xticks[ens], mean_),
                        xytext=(10, 0), textcoords="offset points",
                        fontsize=9, va="center",
                        color="black", fontweight="bold")
        ax.set_xlabel("Calibration")
        ax.set_title(f"({'a' if regime == 'uniform' else 'b'}) "
                     f"{regime}-difficulty benchmarks")
    axes[0].set_ylabel("clean accuracy − contaminated accuracy  (pp)")
    axes[1].set_ylabel("")
    fig.suptitle("Deployment accuracy gap\n"
                 "(positive = wrong direction, negative = correct direction)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig4_inversion.pdf")
    plt.savefig(f"{FIGS}/fig4_inversion.png")
    plt.close()
    print(f"[fig4] saved")


def fig5_lightreg():
    """F1 vs exposure (log-x)."""
    df = pd.read_csv(f"{FIGS}/table5_lightreg.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    pal = sns.color_palette("colorblind", n_colors=df["variant"].nunique())
    for (variant, g), c in zip(df.groupby("variant"), pal):
        g = g.sort_values("epochs")
        axes[0].plot(g["epochs"], g["ensemble_f1"], "o-", color=c, lw=1.5, ms=7,
                     label=f"variant {variant}")
        axes[1].plot(g["epochs"], g["verbatim_rate"], "s-", color=c, lw=1.5, ms=7,
                     label=f"variant {variant}")
    for ax in axes:
        ax.set_xlabel("Fine-tune epochs")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Ensemble F1")
    axes[0].set_ylim(0.8, 1.02)
    axes[0].set_title("(a) Detection F1 vs exposure")
    axes[1].set_ylabel("Verbatim reproduction")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_title("(b) Memorization vs exposure")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig5_lightreg.pdf")
    plt.savefig(f"{FIGS}/fig5_lightreg.png")
    plt.close()
    print(f"[fig5] saved")


def fig6_stratified():
    """Before/after stratification comparison: v2 cells."""
    import os
    sf = f"{PROJECT}/results/stratified_impact/stratified_gaps.csv"
    if not os.path.exists(sf):
        return
    df = pd.read_csv(sf)
    UNIFORM = ["gsm8k_test", "mmlu_test", "hellaswag_val",
                "arc_challenge", "humaneval", "humaneval_sandbox"]
    df["regime"] = df.benchmark.apply(lambda b: "uniform" if b in UNIFORM else "variant")
    v2 = df[df.ensemble == "v2"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pal = {"uniform": sns.color_palette("colorblind")[0],
           "variant": sns.color_palette("colorblind")[3]}
    sns.swarmplot(data=v2, x="regime", y="iv_weighted_gap", hue="regime",
                   size=9, palette=pal, dodge=False, legend=False, ax=ax)
    ax.axhline(0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax.set_ylabel(r"inverse-variance-weighted within-stratum gap")
    ax.set_xlabel("Benchmark regime")
    ax.set_title("v2 calibration with within-difficulty stratified impact analysis\n"
                 "(negative = correct contamination direction)")
    # Annotate mean per regime
    for i, regime in enumerate(["uniform", "variant"]):
        vals = v2[v2.regime == regime].iv_weighted_gap.values
        if len(vals):
            ax.scatter([i], [vals.mean()], marker="D", color="black",
                        s=100, zorder=10, edgecolors="white")
            ax.annotate(f"mean = {vals.mean():+.3f}", (i, vals.mean()),
                        xytext=(10, 0), textcoords="offset points", fontsize=9,
                        fontweight="bold", va="center")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig6_stratified.pdf")
    plt.savefig(f"{FIGS}/fig6_stratified.png")
    plt.close()
    print(f"[fig6] saved")


def main():
    fig1_roc()
    fig2_heatmap()
    fig3_temporal()
    fig4_accuracy_inversion()
    fig5_lightreg()
    fig6_stratified()


if __name__ == "__main__":
    main()
