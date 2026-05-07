#!/usr/bin/env python3
"""Phase 6 — Figures and tables.

Figure 1: ROC curves per detection method on ground truth.
Figure 2: Contamination heatmap (benchmarks × audited models).
Figure 3: Contamination vs model release date.
Figure 4: Reported vs clean accuracy scatter.
Table 1: Detection-method calibration (precision/recall/F1/AUC).
Table 2: Top-10 most contaminated (benchmark, model) cells.
Table 3: Benchmark salvageability classification.
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

PROJECT = "{REPO_ROOT}"
FIGS = f"{PROJECT}/figures"
Path(FIGS).mkdir(parents=True, exist_ok=True)

# Model release dates (approximate) for Figure 3
MODEL_RELEASE = {
    "EleutherAI__pythia-70m": "2023-03",
    "EleutherAI__pythia-410m": "2023-03",
    "EleutherAI__pythia-1b": "2023-03",
    "EleutherAI__pythia-2.8b": "2023-03",
    "EleutherAI__pythia-6.9b": "2023-03",
    "allenai__OLMo-2-1124-7B": "2024-11",
    "meta-llama__Llama-3.1-8B": "2024-07",
    "Qwen__Qwen2.5-7B": "2024-09",
    "deepseek-ai__DeepSeek-V2-Lite": "2024-05",
    "mistralai__Mistral-7B-v0.3": "2024-05",
    "google__gemma-2-9b": "2024-06",
    "microsoft__phi-4": "2024-12",
    "deepseek-ai__DeepSeek-R1-Distill-Qwen-7B": "2025-01",
}


def load_phase3_scores():
    out = {}
    for v in ["A", "B", "C", "D", "E"]:
        p = f"{PROJECT}/results/phase3/{v}/scores.pkl"
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[v] = pickle.load(f)
    return out


def load_calibration():
    p = f"{PROJECT}/results/phase3/calibration.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def load_phase4():
    """Return dict[model_short_name -> scores dict from phase4]."""
    out = {}
    base = f"{PROJECT}/results/phase4"
    if not os.path.exists(base):
        return out
    for d in os.listdir(base):
        p = f"{base}/{d}/scores.pkl"
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[d] = pickle.load(f)
    return out


def load_phase5():
    out = {}
    base = f"{PROJECT}/results/phase5"
    if not os.path.exists(base):
        return out
    for d in os.listdir(base):
        p = f"{base}/{d}/impact.json"
        if os.path.exists(p):
            with open(p) as f:
                out[d] = json.load(f)
    return out


# ---------- Figure 1: ROC per detection method ----------

def figure_1_roc(phase3):
    if not phase3:
        print("[skip fig1] no phase3 scores")
        return

    methods = [k for k in next(iter(phase3.values())).keys() if k != "label"]
    plt.figure(figsize=(7, 6))
    for m in methods:
        ys, ss = [], []
        for v in ["A", "B", "C"]:
            if v not in phase3 or m not in phase3[v]:
                continue
            s = phase3[v][m]; lab = phase3[v]["label"]
            mask = ~np.isnan(s)
            ss.append(s[mask]); ys.append(lab[mask])
        if not ss:
            continue
        s = np.concatenate(ss); y = np.concatenate(ys)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, label=m, lw=1.5, alpha=0.8)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — per detection method (GT variants A–C)")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig1_roc.pdf")
    plt.savefig(f"{FIGS}/fig1_roc.png", dpi=150)
    plt.close()
    print(f"[saved] {FIGS}/fig1_roc.pdf")


# ---------- Figure 2: Contamination heatmap ----------

def figure_2_heatmap(phase4):
    if not phase4:
        print("[skip fig2] no phase4 data")
        return

    rows = []  # (model, benchmark, mean_p_contam)
    for model, data in phase4.items():
        prov = data["provenance"]
        p = data["p_contam"]
        benches = sorted(set(b for b, _ in prov))
        for b in benches:
            mask = np.array([pp[0] == b for pp in prov])
            if mask.sum() == 0 or np.all(np.isnan(p[mask])):
                continue
            rows.append({"model": model.replace("__", "/"), "benchmark": b,
                          "p_contam": float(np.nanmean(p[mask]))})
    df = pd.DataFrame(rows)
    if df.empty:
        print("[skip fig2] empty df")
        return

    mat = df.pivot(index="benchmark", columns="model", values="p_contam")

    plt.figure(figsize=(max(10, 0.7 * len(mat.columns)), 0.5 * len(mat.index) + 3))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1,
                cbar_kws={"label": "mean contamination probability"})
    plt.title("Contamination probability (mean per benchmark × model)")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig2_heatmap.pdf")
    plt.savefig(f"{FIGS}/fig2_heatmap.png", dpi=150)
    plt.close()
    print(f"[saved] {FIGS}/fig2_heatmap.pdf")

    # Also save as CSV
    mat.to_csv(f"{FIGS}/fig2_heatmap_data.csv")


# ---------- Figure 3: Contamination vs release date ----------

def figure_3_temporal(phase4):
    if not phase4:
        print("[skip fig3] no phase4 data")
        return
    rows = []
    for model, data in phase4.items():
        release = MODEL_RELEASE.get(model)
        if not release:
            continue
        mean_p = float(np.nanmean(data["p_contam"])) if not np.all(np.isnan(data["p_contam"])) else None
        if mean_p is None:
            continue
        rows.append({"model": model, "release": release, "p_contam": mean_p})
    if not rows:
        print("[skip fig3] no release-date matches")
        return
    df = pd.DataFrame(rows).sort_values("release")
    plt.figure(figsize=(9, 5))
    plt.scatter(df["release"], df["p_contam"], s=80, alpha=0.8)
    for _, r in df.iterrows():
        plt.annotate(r["model"].split("__")[-1], (r["release"], r["p_contam"]),
                     fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("Model release (YYYY-MM)")
    plt.ylabel("Mean contamination probability")
    plt.title("Benchmark contamination vs. model release date")
    plt.xticks(rotation=30)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig3_temporal.pdf")
    plt.savefig(f"{FIGS}/fig3_temporal.png", dpi=150)
    plt.close()
    print(f"[saved] {FIGS}/fig3_temporal.pdf")


# ---------- Figure 4: Reported vs clean accuracy ----------

def figure_4_accuracy_scatter(phase5):
    if not phase5:
        print("[skip fig4] no phase5 data")
        return
    rows = []
    for model, by_bench in phase5.items():
        for bench, subs in by_bench.items():
            full = (subs.get("full") or {}).get("acc")
            clean = (subs.get("clean") or {}).get("acc")
            if full is None or clean is None:
                continue
            rows.append({"model": model, "benchmark": bench, "full": full, "clean": clean})
    if not rows:
        print("[skip fig4] no rows")
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 6))
    for bench, sub in df.groupby("benchmark"):
        plt.scatter(sub["full"], sub["clean"], label=bench, s=60, alpha=0.8)
    lim = [0, max(df["full"].max(), df["clean"].max()) * 1.05]
    plt.plot(lim, lim, "k--", alpha=0.3, lw=0.8, label="y = x")
    plt.xlabel("Accuracy on full benchmark")
    plt.ylabel("Accuracy on clean subset (p_contam < 0.1)")
    plt.title("Reported vs. clean accuracy — contamination inflates scores")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.2)
    plt.xlim(lim); plt.ylim(lim)
    plt.tight_layout()
    plt.savefig(f"{FIGS}/fig4_accuracy.pdf")
    plt.savefig(f"{FIGS}/fig4_accuracy.png", dpi=150)
    plt.close()
    print(f"[saved] {FIGS}/fig4_accuracy.pdf")


# ---------- Tables ----------

def table_1_calibration(calibration):
    if not calibration:
        return
    rows = []
    for m, d in calibration.get("per_method", {}).items():
        rows.append({"method": m, "threshold": d["threshold"],
                     "dev_F1": d["dev"]["f1"], "dev_AUC": d["dev"].get("auc"),
                     "val_F1": (d["val"] or {}).get("f1"),
                     "val_AUC": (d["val"] or {}).get("auc")})
    df = pd.DataFrame(rows)
    df.to_csv(f"{FIGS}/table1_calibration.csv", index=False)
    with open(f"{FIGS}/table1_calibration.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))
    print(f"[saved] {FIGS}/table1_calibration.csv")


def table_2_top_contaminated(phase4, top=10):
    rows = []
    for model, data in phase4.items():
        prov = data["provenance"]; p = data["p_contam"]
        by_b = {}
        for (b, _i), pp in zip(prov, p):
            if np.isnan(pp): continue
            by_b.setdefault(b, []).append(pp)
        for b, ps in by_b.items():
            rows.append({"model": model.replace("__", "/"), "benchmark": b,
                          "mean_p_contam": float(np.mean(ps)),
                          "p90_p_contam": float(np.percentile(ps, 90)),
                          "high_risk_rate": float(np.mean(np.array(ps) > 0.5)),
                          "n_items": len(ps)})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("mean_p_contam", ascending=False).head(top)
    df.to_csv(f"{FIGS}/table2_top_contaminated.csv", index=False)
    with open(f"{FIGS}/table2_top_contaminated.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))
    print(f"[saved] {FIGS}/table2_top_contaminated.csv")


def table_3_benchmark_salvageability(phase4):
    rows = []
    by_bench = {}
    for model, data in phase4.items():
        prov = data["provenance"]; p = data["p_contam"]
        for (b, _i), pp in zip(prov, p):
            if np.isnan(pp): continue
            by_bench.setdefault(b, []).append(pp)
    for b, ps in by_bench.items():
        ps = np.array(ps)
        mean_p = float(ps.mean())
        high_rate = float((ps > 0.5).mean())
        if high_rate < 0.05:
            label = "clean"
        elif high_rate < 0.25:
            label = "salvageable"
        else:
            label = "compromised"
        rows.append({"benchmark": b, "n_model_item_pairs": len(ps),
                      "mean_p_contam": mean_p, "high_risk_rate": high_rate,
                      "classification": label})
    df = pd.DataFrame(rows).sort_values("mean_p_contam", ascending=False)
    df.to_csv(f"{FIGS}/table3_salvageability.csv", index=False)
    with open(f"{FIGS}/table3_salvageability.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))
    print(f"[saved] {FIGS}/table3_salvageability.csv")


def table_4_accuracy_drops(phase5):
    """Full vs. clean vs. contaminated accuracy per (model, benchmark)."""
    if not phase5:
        return
    rows = []
    for model, by_bench in phase5.items():
        for bench, subs in by_bench.items():
            full_m = subs.get("full") or {}
            clean_m = subs.get("clean") or {}
            contam_m = subs.get("contaminated") or {}
            rows.append({
                "model": model.replace("__", "/"),
                "benchmark": bench,
                "full_acc": full_m.get("acc"),
                "full_n": full_m.get("n"),
                "clean_acc": clean_m.get("acc"),
                "clean_n": clean_m.get("n"),
                "contam_acc": contam_m.get("acc"),
                "contam_n": contam_m.get("n"),
                "full_minus_clean": (full_m.get("acc") or 0) - (clean_m.get("acc") or 0)
                    if full_m.get("acc") is not None and clean_m.get("acc") is not None else None,
                "clean_minus_contam": (clean_m.get("acc") or 0) - (contam_m.get("acc") or 0)
                    if clean_m.get("acc") is not None and contam_m.get("acc") is not None else None,
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{FIGS}/table4_accuracy_drops.csv", index=False)
    with open(f"{FIGS}/table4_accuracy_drops.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))
    print(f"[saved] {FIGS}/table4_accuracy_drops.csv")


def main():
    phase3 = load_phase3_scores()
    calibration = load_calibration()
    phase4 = load_phase4()
    phase5 = load_phase5()

    print(f"[data] phase3={len(phase3)}, phase4={len(phase4)}, phase5={len(phase5)}")

    figure_1_roc(phase3)
    figure_2_heatmap(phase4)
    figure_3_temporal(phase4)
    figure_4_accuracy_scatter(phase5)
    table_1_calibration(calibration)
    table_2_top_contaminated(phase4)
    table_3_benchmark_salvageability(phase4)
    table_4_accuracy_drops(phase5)


if __name__ == "__main__":
    main()
