#!/usr/bin/env python3
"""Export paper artifacts in safe, portable formats.

Replaces pickle with JSON + numpy npz where possible; computes SHA-256
hashes for release-integrity; generates a croissant metadata file.

Addresses reviewer R3's D&B critique about pickle security / missing
persistent identifiers / no croissant metadata.
"""
import hashlib
import json
import os
import pickle
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

PROJECT = "{REPO_ROOT}"
OUT = f"{PROJECT}/release_artifacts"
Path(OUT).mkdir(parents=True, exist_ok=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_phase4_scores():
    """For each model in phase4_v2, export scores.pkl as a JSON-portable
    dict with per-item p_contam and provenance."""
    for phase_dir, subdir in [
        (f"{PROJECT}/results/phase4", "phase4_v1"),
        (f"{PROJECT}/results/phase4_v2", "phase4_v2"),
    ]:
        out_dir = f"{OUT}/{subdir}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for model in sorted(os.listdir(phase_dir)):
            if model.startswith("_"):
                continue
            sp = f"{phase_dir}/{model}/scores.pkl"
            if not os.path.exists(sp):
                continue
            try:
                with open(sp, "rb") as f:
                    d = pickle.load(f)
            except Exception as e:
                print(f"[skip] {sp}: {e}")
                continue
            # Save provenance + p_contam as a simple JSON + npz
            prov = [list(p) for p in d["provenance"]]
            np.savez_compressed(f"{out_dir}/{model}__p_contam.npz",
                                p_contam=d["p_contam"],
                                provenance_benchmark=[p[0] for p in prov],
                                provenance_item_id=[p[1] for p in prov])
            print(f"  exported {subdir}/{model}")


def export_ensembles():
    """Export ensemble pickles as JSON coefficient dumps (+ keep pickle with SHA)."""
    # Ensembles are in results/phase3* directories
    candidates = [
        f"{PROJECT}/results/phase3/ensemble_v1.pkl",
        f"{PROJECT}/results/phase3_v2/ensemble_v2.pkl",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            clf = pickle.load(f)
        name = os.path.basename(path).replace(".pkl", "")
        out = {}
        # sklearn LogisticRegression, LGBMClassifier, etc.
        if hasattr(clf, "coef_"):
            out["type"] = type(clf).__name__
            out["coef"] = clf.coef_.tolist()
            out["intercept"] = clf.intercept_.tolist()
            if hasattr(clf, "feature_names_in_"):
                out["features"] = list(clf.feature_names_in_)
        elif hasattr(clf, "booster_"):
            # LightGBM
            out["type"] = type(clf).__name__
            out["n_features"] = int(clf.n_features_)
            try:
                out["model_json"] = clf.booster_.dump_model()
            except Exception:
                out["note"] = "could not dump booster"
        else:
            out["type"] = type(clf).__name__
            out["note"] = "custom ensemble; not JSON-serializable"
        json.dump(out, open(f"{OUT}/{name}.json", "w"), indent=2, default=str)
        print(f"  exported ensemble {name} as JSON")


def compute_hashes():
    """Compute SHA-256 for key release artifacts."""
    # Key artifacts to hash
    targets = [
        (f"{PROJECT}/results/phase3_v2/ensemble_v2.pkl", "ensemble_v2.pkl"),
        (f"{PROJECT}/results/phase3/ensemble_v1.pkl", "ensemble_v1.pkl"),
        (f"{PROJECT}/paper/main.tex", "paper/main.tex"),
        (f"{PROJECT}/paper/pre_registration.md", "paper/pre_registration.md"),
    ]
    hashes = {}
    for path, name in targets:
        if os.path.exists(path):
            hashes[name] = sha256_of(path)
    # Also per-model JSON exports we just wrote
    for phase in ["phase4_v1", "phase4_v2"]:
        d = f"{OUT}/{phase}"
        if not os.path.exists(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz"):
                hashes[f"{phase}/{f}"] = sha256_of(f"{d}/{f}")
    with open(f"{OUT}/SHA256SUMS.txt", "w") as f:
        for name, h in sorted(hashes.items()):
            f.write(f"{h}  {name}\n")
    print(f"Wrote {len(hashes)} hashes to {OUT}/SHA256SUMS.txt")
    return hashes


def write_croissant(hashes):
    """Write a minimal croissant metadata document.

    Croissant is the ML dataset metadata standard. See
    https://docs.mlcommons.org/croissant/docs/croissant-spec.html
    """
    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
        },
        "@type": "sc:Dataset",
        "@id": "https://example.com/datasets/paper5-contamination-db",
        "name": "paper5-contamination-db",
        "description": "Per-example LLM contamination scores for 14-18 open-weight models across 13 benchmarks under two calibration regimes (v1: cross-benchmark negatives [deployment-invalid, replication-only]; v2: same-benchmark negatives [deployment-valid]).",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": "https://example.com/datasets/paper5-contamination-db",
        "version": "1.0-neurips2026",
        "datePublished": datetime.now(timezone.utc).isoformat(),
        "creator": {
            "@type": "Organization",
            "name": "anonymous-for-review",
        },
        "citation": "Anonymous, 'The Calibration Mismatch in LLM Contamination Detection', NeurIPS 2026 D&B submission.",
        "distribution": [
            {"@type": "cr:FileObject", "@id": "ensemble_v2",
              "contentUrl": "ensemble_v2.pkl",
              "sha256": hashes.get("ensemble_v2.pkl", "TBD"),
              "encodingFormat": "application/python-pickle",
              "description": "v2 (same-benchmark negatives) calibrated ensemble — safe for deployment."},
            {"@type": "cr:FileObject", "@id": "ensemble_v1",
              "contentUrl": "ensemble_v1.pkl",
              "sha256": hashes.get("ensemble_v1.pkl", "TBD"),
              "encodingFormat": "application/python-pickle",
              "description": "v1 (cross-benchmark negatives) calibrated ensemble — REPLICATION ONLY, DO NOT DEPLOY. Exhibits domain-confound inversion at deployment."},
            {"@type": "cr:FileSet", "@id": "v2_per_model_scores",
              "description": "Per-example p_contam scores under v2 calibration for each audited model.",
              "encodingFormat": "application/octet-stream",
              "includes": "phase4_v2/*__p_contam.npz"},
            {"@type": "cr:FileSet", "@id": "v1_per_model_scores",
              "description": "Per-example p_contam scores under v1 calibration (replication-only).",
              "encodingFormat": "application/octet-stream",
              "includes": "phase4_v1/*__p_contam.npz"},
        ],
        "rai:dataBiases": "v1 scores exhibit a systematic domain-benchmark confound; see paper §4-5. v2 scores are the recommended deployment target. Scores are uncalibrated against closed-API LLMs (GPT-4/Claude/Gemini); methodology is demonstrated on open-weight models only.",
        "rai:dataUseCases": "Intended for: (a) replication of the v1/v2 calibration-mismatch finding; (b) benchmark-audit workflows that feed per-example scores into within-difficulty-stratified accuracy-gap tests (§5.2 of the paper). Not intended for: hard thresholding of individual items without the accompanying accuracy-gap validation.",
        "rai:personalSensitiveInformation": "No PII. All benchmark items are from public evaluation datasets with open licenses.",
    }
    with open(f"{OUT}/croissant.json", "w") as f:
        json.dump(croissant, f, indent=2)
    print(f"Wrote croissant metadata to {OUT}/croissant.json")


def main():
    print("Exporting phase4 scores as JSON+npz...")
    export_phase4_scores()
    print("\nExporting ensemble JSON dumps...")
    export_ensembles()
    print("\nComputing SHA-256 hashes...")
    hashes = compute_hashes()
    print("\nWriting croissant metadata...")
    write_croissant(hashes)
    print("\nDone.")


if __name__ == "__main__":
    main()
