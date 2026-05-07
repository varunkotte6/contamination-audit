#!/usr/bin/env python3
"""Analyze reference-model sensitivity by recomputing refmodel_delta
with three alternative reference models, then re-applying the v2
ensemble to Llama-3.1-8B on MMLU and ARC (the two Bonferroni-surviving
cells) and testing whether the correct-direction contamination signal
persists.
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT = "{REPO_ROOT}"
LLAMA_SCORES = f"{PROJECT}/results/phase4/meta-llama__Llama-3.1-8B/scores.pkl"
REFSENS_DIR = f"{PROJECT}/results/refmodel_sensitivity"
PHASE5_LLAMA = f"{PROJECT}/results/phase5_full/meta-llama__Llama-3.1-8B/per_item.pkl"
OUT = f"{PROJECT}/results/refmodel_sensitivity"
Path(OUT).mkdir(parents=True, exist_ok=True)

ALT_REFS = ["pythia69b", "olmo27b", "llama318b"]
BENCHES = ["mmlu_test", "arc_challenge", "gsm8k_test"]
N_ITEMS = 500  # same as refsens run


def main():
    # Load Llama phase4 scores (features + provenance)
    with open(LLAMA_SCORES, "rb") as f:
        d = pickle.load(f)
    prov = d["provenance"]
    feats = d["features"]
    loss_ft = feats["loss_ft"]   # Llama-3.1-8B per-item loss (audited)
    loss_ref_pythia1b = feats["loss_base"]
    refmodel_delta_orig = feats["refmodel_delta"]

    # Per-item correlation / deployment-direction analysis
    print("\n=== Reference-model sensitivity ===")
    results = []
    for bench in BENCHES:
        idxs = [i for i, (b, _) in enumerate(prov) if b == bench]
        idxs = idxs[:N_ITEMS]
        if len(idxs) < 10:
            continue
        la = loss_ft[idxs]
        lr_orig = loss_ref_pythia1b[idxs]
        rmd_orig = refmodel_delta_orig[idxs]

        row = {"benchmark": bench, "n": len(idxs),
                "rmd_orig_mean": float(np.nanmean(rmd_orig)),
                "rmd_orig_std": float(np.nanstd(rmd_orig))}

        for ref in ALT_REFS:
            npz = np.load(f"{REFSENS_DIR}/refloss_{ref}.npz")
            if bench not in npz.files:
                continue
            lr_alt = npz[bench][:len(idxs)]
            # Align lengths (refsens used first 500 from jsonl; phase4 provenance
            # matches this since both load with the same order)
            n = min(len(la), len(lr_alt))
            la_n = la[:n]; lr_alt_n = lr_alt[:n]; rmd_orig_n = rmd_orig[:n]
            rmd_alt = lr_alt_n - la_n
            # Pearson on refmodel_delta
            m = ~(np.isnan(rmd_alt) | np.isnan(rmd_orig_n))
            if m.sum() < 10:
                continue
            r_pearson = float(np.corrcoef(rmd_alt[m], rmd_orig_n[m])[0, 1])
            r_spearman, _ = stats.spearmanr(rmd_alt[m], rmd_orig_n[m])
            row[f"rmd_{ref}_mean"] = float(np.nanmean(rmd_alt))
            row[f"rmd_{ref}_std"] = float(np.nanstd(rmd_alt))
            row[f"pearson_{ref}"] = r_pearson
            row[f"spearman_{ref}"] = float(r_spearman)
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(f"{OUT}/analysis.csv", index=False)
    print("\n=== Summary: refmodel_delta with Pythia-1B vs alternative refs ===")
    print(f"{'benchmark':<18s} {'n':>4s} "
          f"{'r_ρ(py69b)':>12s} {'r_ρ(olmo)':>12s} {'r_ρ(llama)':>12s}")
    for _, r in df.iterrows():
        print(f"{r['benchmark']:<18s} {int(r['n']):>4d} "
              f"{r.get('pearson_pythia69b', np.nan):>12.4f} "
              f"{r.get('pearson_olmo27b', np.nan):>12.4f} "
              f"{r.get('pearson_llama318b', np.nan):>12.4f}")

    # Deployment-direction test: for each alt ref, use rmd_alt instead of
    # rmd_orig as the p_contam proxy, split items into clean (low rmd_alt)
    # and contaminated (high rmd_alt), and see if the accuracy gap still
    # goes in the correct direction (clean > contam).
    # Simplification: use rmd_alt alone (not full ensemble) as a proxy
    # ranking. This is a non-ensembled sensitivity check; if it still
    # produces correct-direction gaps for Llama's MMLU/ARC, the ensemble
    # result is robust to reference choice.
    print("\n=== Deployment-direction persistence check ===")
    print("(Using refmodel_delta alone as p_contam proxy;")
    print(" clean = below-median rmd, contam = above-median rmd)")
    try:
        with open(PHASE5_LLAMA, "rb") as f:
            p5 = pickle.load(f)
    except Exception as e:
        print(f"[WARN] can't load phase5 per-item: {e}")
        return

    for bench in ["mmlu_test", "arc_challenge"]:
        if bench not in p5:
            continue
        correct = np.array([r["correct"] for r in p5[bench]["per_item"]],
                            dtype=np.int32)
        idxs = [i for i, (b, _) in enumerate(prov) if b == bench]
        idxs = idxs[:N_ITEMS]
        n = min(len(correct), len(idxs))
        correct = correct[:n]
        la = loss_ft[idxs][:n]

        for ref_name, ref_key in [("Pythia-1B (orig)", None),
                                    ("Pythia-6.9B", "pythia69b"),
                                    ("OLMo-2-7B", "olmo27b"),
                                    ("Llama-3.1-8B base", "llama318b")]:
            if ref_key is None:
                rmd = refmodel_delta_orig[idxs][:n]
            else:
                npz = np.load(f"{REFSENS_DIR}/refloss_{ref_key}.npz")
                lr_alt = npz[bench][:n]
                rmd = lr_alt - la

            m = ~(np.isnan(rmd) | np.isnan(correct))
            if m.sum() < 20:
                continue
            rmd_m = rmd[m]; correct_m = correct[m]
            median = np.nanmedian(rmd_m)
            # Higher refmodel_delta = more memorized (audited model has lower loss
            # than reference, so delta = loss_ref - loss_audit is LARGER for
            # memorized). So "contam" = high rmd, "clean" = low rmd.
            clean = correct_m[rmd_m < median]
            contam = correct_m[rmd_m >= median]
            gap = float(clean.mean() - contam.mean())
            n_cl = int(len(clean)); n_ct = int(len(contam))
            # Permutation p
            obs = gap
            combined = np.concatenate([clean, contam])
            rng = np.random.default_rng(0)
            n_perm = 5000
            ex = 0
            for _ in range(n_perm):
                rng.shuffle(combined)
                if (combined[:n_cl].mean() - combined[n_cl:].mean()) <= obs:
                    ex += 1
            p_perm = ex / n_perm
            print(f"  {bench:<15s} ref={ref_name:<22s} "
                  f"clean={clean.mean():.3f}(n={n_cl}) "
                  f"contam={contam.mean():.3f}(n={n_ct}) "
                  f"gap={gap:+.3f} p_one={p_perm:.4f}")


if __name__ == "__main__":
    main()
