#!/usr/bin/env python3
"""W4: reference-model robustness check for the 7 BH-FDR cells.

The reviewer asks whether the 7 BH-FDR correct-direction v2 cells
survive when refmodel_delta is computed with a different reference
(Pythia-6.9B or OLMo-2-7B instead of Pythia-1B). A full retraining of
the ensemble with alternative reference losses at calibration time
would require computing calibration-item losses on each new reference,
which we have not run. A cheaper check: at audit time only, replace
loss_base (Pythia-1B) with the alternative reference loss and apply
the existing v2 ensemble. This tests whether the deployment signal
would transfer if the reference model family were changed, holding
the ensemble weights fixed.

For each of the 7 BH-FDR cells where we have the alternative reference
loss available, we replace loss_base in the feature vector and rerun
the v2 ensemble prediction, then recompute the stratified gap.
"""
import os
import pickle
import numpy as np

PROJECT = "{REPO_ROOT}"
PHASE4 = f"{PROJECT}/results/phase4"
PHASE4_V2 = f"{PROJECT}/results/phase4_v2"
PHASE5 = f"{PROJECT}/results/phase5_full"
ENSEMBLE_V2 = f"{PROJECT}/results/phase3_v2/ensemble.pkl"
REFMOD_DIR = f"{PROJECT}/results/refmodel_sensitivity"

# 7 Bonferroni-surviving correct-direction v2 cells from Section 5.4
BH_CELLS = [
    ("meta-llama__Llama-3.1-8B", "mmlu_test"),
    ("Qwen__Qwen2.5-32B", "mmlu_test"),
    ("meta-llama__Llama-3.1-8B", "arc_challenge"),
    ("Qwen__Qwen2.5-32B", "arc_challenge"),
    ("Qwen__Qwen2.5-14B", "arc_challenge"),
    ("allenai__OLMo-2-1124-7B", "gsm8k_test"),
    # Mistral × HumanEval-sandbox: has sandbox variant, skip for now
]

V2_FEATURES = ["ngram13_pile", "ngram13_fineweb",
                "mink_5", "mink_10", "mink_20", "mink_30", "mink_50",
                "loss_ft", "loss_base", "refmodel_delta"]


def load_alt_refloss(ref_name):
    p = f"{REFMOD_DIR}/refloss_{ref_name}.npz"
    if not os.path.exists(p):
        return None
    return np.load(p)


def load_cell(model, bench):
    sp = f"{PHASE4}/{model}/scores.pkl"
    if not os.path.exists(sp):
        return None
    d = pickle.load(open(sp, "rb"))
    idxs = [i for i, (b, _) in enumerate(d["provenance"]) if b == bench]
    if not idxs:
        return None
    item_ids = np.array([d["provenance"][i][1] for i in idxs])
    order = np.argsort(item_ids)
    feats = {k: v[idxs][order] for k, v in d["features"].items()}
    return feats, item_ids[order]


def load_correct(model, bench):
    # variant-difficulty benches use different dirs; skip for now — all BH
    # cells are on uniform-difficulty benches in phase5_full
    p = f"{PHASE5}/{model}/per_item.pkl"
    if not os.path.exists(p):
        return None
    d = pickle.load(open(p, "rb"))
    if bench not in d:
        return None
    return np.array([x["correct"] for x in d[bench]["per_item"]], dtype=float)


def main():
    ens = pickle.load(open(ENSEMBLE_V2, "rb"))
    scaler = ens["scaler"]; clf = ens["clf"]
    print("v2 ensemble methods:", ens["methods"])

    # Report per-cell stats under original (Pythia-1B ref) vs alt reference
    print("\nW4 sensitivity test: swap loss_base (Pythia-1B) with alt ref at "
          "audit time, keeping v2 ensemble weights fixed.\n")

    for ref_name, label in [("pythia69b", "Pythia-6.9B"),
                              ("olmo27b", "OLMo-2-7B")]:
        alt = load_alt_refloss(ref_name)
        if alt is None:
            print(f"[skip] {label}: refloss not cached")
            continue
        print(f"--- Alt reference: {label} ---")
        print(f"{'model':<40s} {'bench':<16s} {'orig_gap':>8s} "
              f"{'orig_p':>7s} {'alt_gap':>8s} {'alt_p':>7s}")
        for model, bench in BH_CELLS:
            if bench not in alt.files:
                continue
            feats_tup = load_cell(model, bench)
            correct = load_correct(model, bench)
            if feats_tup is None or correct is None:
                continue
            feats, item_ids = feats_tup
            n_alt = len(alt[bench])
            n = min(len(feats["loss_base"]), n_alt, len(correct))
            # Cap to n items (alt refloss only covers first 500 per bench)
            feats_n = {k: v[:n] for k, v in feats.items()}
            correct = correct[:n]
            # Original ensemble output (= existing phase4_v2 score), restricted
            orig_p_path = f"{PHASE4_V2}/{model}/scores.pkl"
            orig_d = pickle.load(open(orig_p_path, "rb"))
            orig_idx = [i for i, (b, _) in enumerate(orig_d["provenance"])
                        if b == bench][:n]
            orig_p = orig_d["p_contam"][orig_idx]

            # Substitute loss_base with alt reference loss, recompute
            # refmodel_delta = loss_base - loss_ft (where loss_ft = audited)
            alt_loss_base = alt[bench][:n]
            feats_alt = dict(feats_n)
            feats_alt["loss_base"] = alt_loss_base
            feats_alt["refmodel_delta"] = alt_loss_base - feats_n["loss_ft"]
            X_alt = np.column_stack([feats_alt[f] for f in V2_FEATURES])
            X_alt = np.nan_to_num(X_alt)
            alt_p = clf.predict_proba(scaler.transform(X_alt))[:, 1]
            # Stratified gap (unstratified for simplicity here)
            def gap(pp):
                valid = ~np.isnan(pp)
                c = correct[valid]; p = pp[valid]
                clean = p < 0.1; contam = p > 0.5
                if clean.sum() < 10 or contam.sum() < 10:
                    return None, None
                cc = c[clean].mean(); ct = c[contam].mean()
                # Permutation test
                rng = np.random.default_rng(42)
                obs = cc - ct
                cc_arr = c[clean]; ct_arr = c[contam]
                comb = np.concatenate([cc_arr, ct_arr])
                nle = 0
                for _ in range(2000):
                    rng.shuffle(comb)
                    if comb[:len(cc_arr)].mean() - comb[len(cc_arr):].mean() <= obs:
                        nle += 1
                return float(obs), nle / 2000
            og, op = gap(orig_p)
            ag, ap = gap(alt_p)
            og_s = f"{og:+.3f}" if og is not None else "  n/a"
            op_s = f"{op:.3f}" if op is not None else "  n/a"
            ag_s = f"{ag:+.3f}" if ag is not None else "  n/a"
            ap_s = f"{ap:.3f}" if ap is not None else "  n/a"
            print(f"{model:<40s} {bench:<16s} {og_s:>8s} {op_s:>7s} "
                   f"{ag_s:>8s} {ap_s:>7s}")


if __name__ == "__main__":
    main()
