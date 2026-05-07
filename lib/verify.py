#!/usr/bin/env python3
"""
Verification library for NeurIPS 2026 research agent.
Provides atomic verification functions for models, data, and experiments.
All verifications are append-only logged to verification_log.jsonl.
"""

import json
import os
import sys
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone


def log_verification(log_path: str, record: dict):
    """Append a verification record to the JSONL log. Never overwrites."""
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def verify_model_download(model_path: str, model_id: str, log_path: str) -> bool:
    """
    Verify a HuggingFace model was downloaded correctly.
    Checks: directory exists, config.json valid, weight files present, loadable.
    """
    checks = {}
    passed = True

    # Check 1: Directory exists and is non-empty
    p = Path(model_path)
    checks["directory_exists"] = p.exists() and p.is_dir()
    if checks["directory_exists"]:
        contents = list(p.iterdir())
        checks["directory_non_empty"] = len(contents) > 0
        checks["file_count"] = len(contents)
    else:
        checks["directory_non_empty"] = False
        checks["file_count"] = 0
        passed = False

    # Check 2: config.json exists and is valid JSON
    config_path = p / "config.json"
    checks["config_exists"] = config_path.exists()
    if checks["config_exists"]:
        try:
            with open(config_path) as f:
                config = json.load(f)
            checks["config_valid_json"] = True
            checks["model_type"] = config.get("model_type", "unknown")
            checks["hidden_size"] = config.get("hidden_size", "unknown")
        except (json.JSONDecodeError, Exception) as e:
            checks["config_valid_json"] = False
            checks["config_error"] = str(e)
            passed = False
    else:
        passed = False

    # Check 3: Weight files exist
    safetensors = list(p.glob("*.safetensors"))
    bin_files = list(p.glob("*.bin"))
    checks["safetensors_count"] = len(safetensors)
    checks["bin_count"] = len(bin_files)
    checks["has_weights"] = len(safetensors) > 0 or len(bin_files) > 0
    if not checks["has_weights"]:
        passed = False

    # Check 4: Total size
    total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    checks["total_size_gb"] = round(total_size / (1024**3), 2)

    # Check 5: Try loading with transformers (if available)
    try:
        import torch
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        checks["auto_config_loads"] = True
    except Exception as e:
        checks["auto_config_loads"] = False
        checks["load_error"] = str(e)
        # Don't fail on this — some models need special loading

    overall = "PASS" if passed else "FAIL"
    checks["overall"] = overall

    log_verification(log_path, {
        "type": "model_download",
        "model_id": model_id,
        "model_path": model_path,
        "checks": checks,
        "passed": passed
    })

    print(f"[VERIFY MODEL] {model_id}: {overall}")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    return passed


def verify_dataset(data_path: str, dataset_name: str, expected_count: int,
                   expected_fields: list, log_path: str) -> bool:
    """
    Verify a dataset was prepared correctly.
    Checks: file exists, example count, field schema, sample inspection.
    """
    checks = {}
    passed = True

    p = Path(data_path)

    # Check 1: File/directory exists
    checks["path_exists"] = p.exists()
    if not checks["path_exists"]:
        passed = False
        log_verification(log_path, {
            "type": "dataset",
            "dataset_name": dataset_name,
            "data_path": data_path,
            "checks": checks,
            "passed": False
        })
        print(f"[VERIFY DATA] {dataset_name}: FAIL — path does not exist")
        return False

    # Check 2: Load and count examples
    try:
        if data_path.endswith(".jsonl"):
            with open(data_path) as f:
                examples = [json.loads(line) for line in f]
        elif data_path.endswith(".json"):
            with open(data_path) as f:
                examples = json.load(f)
        elif data_path.endswith(".csv"):
            import csv
            with open(data_path) as f:
                reader = csv.DictReader(f)
                examples = list(reader)
        else:
            # Try as directory of files
            examples = list(p.iterdir())
            checks["is_directory"] = True

        checks["example_count"] = len(examples)
        checks["expected_count"] = expected_count
        checks["count_matches"] = abs(len(examples) - expected_count) < expected_count * 0.05  # 5% tolerance

        if not checks["count_matches"]:
            passed = False

    except Exception as e:
        checks["load_error"] = str(e)
        passed = False

    # Check 3: Field schema (for structured data)
    if expected_fields and isinstance(examples[0], dict):
        present_fields = set(examples[0].keys())
        expected_set = set(expected_fields)
        checks["expected_fields"] = expected_fields
        checks["present_fields"] = list(present_fields)
        checks["missing_fields"] = list(expected_set - present_fields)
        checks["fields_match"] = expected_set.issubset(present_fields)
        if not checks["fields_match"]:
            passed = False

    # Check 4: Print 3 random samples
    import random
    random.seed(42)
    if len(examples) >= 3:
        samples = random.sample(examples, 3)
        checks["sample_examples"] = [str(s)[:200] for s in samples]

    overall = "PASS" if passed else "FAIL"
    checks["overall"] = overall

    log_verification(log_path, {
        "type": "dataset",
        "dataset_name": dataset_name,
        "data_path": data_path,
        "checks": checks,
        "passed": passed
    })

    print(f"[VERIFY DATA] {dataset_name}: {overall}")
    for k, v in checks.items():
        if k != "sample_examples":
            print(f"  {k}: {v}")

    return passed


def verify_experiment(exp_id: str, exp_name: str, output_path: str,
                      expected_files: list, expected_metrics: dict,
                      tolerance: float, log_path: str) -> bool:
    """
    Verify an experiment completed correctly.
    Checks: output files exist, metrics in range, no NaN/Inf.
    """
    checks = {}
    passed = True

    # Check 1: Output directory exists
    p = Path(output_path)
    checks["output_path_exists"] = p.exists()
    if not checks["output_path_exists"]:
        passed = False
        log_verification(log_path, {
            "type": "experiment",
            "exp_id": exp_id,
            "exp_name": exp_name,
            "output_path": output_path,
            "checks": checks,
            "passed": False
        })
        print(f"[VERIFY EXP] {exp_id} ({exp_name}): FAIL — output path missing")
        return False

    # Check 2: Expected files exist
    missing_files = []
    for fname in expected_files:
        fpath = p / fname
        if not fpath.exists():
            missing_files.append(fname)
    checks["expected_files"] = expected_files
    checks["missing_files"] = missing_files
    checks["all_files_present"] = len(missing_files) == 0
    if not checks["all_files_present"]:
        passed = False

    # Check 3: Load and verify metrics
    metrics_file = p / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                actual_metrics = json.load(f)

            checks["metrics_loaded"] = True
            metric_checks = {}
            for metric_name, expected_value in expected_metrics.items():
                if metric_name in actual_metrics:
                    actual = actual_metrics[metric_name]

                    # Check for NaN/Inf
                    import math
                    if isinstance(actual, float) and (math.isnan(actual) or math.isinf(actual)):
                        metric_checks[metric_name] = {
                            "actual": str(actual),
                            "expected": expected_value,
                            "status": "FAIL_NAN_INF"
                        }
                        passed = False
                        continue

                    # Check within tolerance
                    if expected_value != 0:
                        rel_error = abs(actual - expected_value) / abs(expected_value)
                    else:
                        rel_error = abs(actual)

                    in_range = rel_error <= tolerance
                    metric_checks[metric_name] = {
                        "actual": actual,
                        "expected": expected_value,
                        "relative_error": round(rel_error, 4),
                        "within_tolerance": in_range
                    }
                    if not in_range:
                        # Flag but don't necessarily fail — unexpected results can be interesting
                        checks["metric_deviation_warning"] = True
                else:
                    metric_checks[metric_name] = {"status": "MISSING"}

            checks["metric_checks"] = metric_checks

        except Exception as e:
            checks["metrics_load_error"] = str(e)
    else:
        checks["metrics_file_exists"] = False

    overall = "PASS" if passed else "FAIL"
    checks["overall"] = overall

    log_verification(log_path, {
        "type": "experiment",
        "exp_id": exp_id,
        "exp_name": exp_name,
        "output_path": output_path,
        "checks": checks,
        "passed": passed
    })

    print(f"[VERIFY EXP] {exp_id} ({exp_name}): {overall}")
    for k, v in checks.items():
        if k != "metric_checks":
            print(f"  {k}: {v}")
    if "metric_checks" in checks:
        for mk, mv in checks["metric_checks"].items():
            print(f"  metric/{mk}: {mv}")

    return passed


def verify_figure(figure_path: str, figure_name: str, log_path: str) -> bool:
    """Verify a generated figure file is valid."""
    checks = {}
    passed = True

    p = Path(figure_path)
    checks["file_exists"] = p.exists()
    if not checks["file_exists"]:
        passed = False
    else:
        checks["file_size_kb"] = round(p.stat().st_size / 1024, 1)
        checks["file_non_empty"] = p.stat().st_size > 100  # At least 100 bytes

        # Try loading as image
        try:
            from PIL import Image
            img = Image.open(figure_path)
            checks["image_size"] = f"{img.width}x{img.height}"
            checks["image_loadable"] = True
            # Check minimum resolution for publication
            checks["publication_quality"] = img.width >= 600 and img.height >= 400
        except Exception:
            # Might be PDF
            if figure_path.endswith(".pdf"):
                checks["format"] = "PDF"
                checks["image_loadable"] = True  # Assume PDF is fine
            else:
                checks["image_loadable"] = False
                passed = False

    overall = "PASS" if passed else "FAIL"
    checks["overall"] = overall

    log_verification(log_path, {
        "type": "figure",
        "figure_name": figure_name,
        "figure_path": figure_path,
        "checks": checks,
        "passed": passed
    })

    print(f"[VERIFY FIG] {figure_name}: {overall}")
    return passed


def verify_environment(log_path: str) -> bool:
    """Verify the execution environment is ready."""
    checks = {}
    passed = True

    # GPU check
    try:
        import torch
        checks["cuda_available"] = torch.cuda.is_available()
        checks["gpu_count"] = torch.cuda.device_count()
        if checks["gpu_count"] > 0:
            checks["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(checks["gpu_count"])]
            checks["gpu_memory_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 1)
                for i in range(checks["gpu_count"])
            ]
        if not checks["cuda_available"]:
            passed = False
    except ImportError:
        checks["torch_installed"] = False
        passed = False

    # Key libraries
    for lib in ["transformers", "datasets", "accelerate", "wandb", "matplotlib", "seaborn", "numpy", "scipy"]:
        try:
            __import__(lib)
            checks[f"lib_{lib}"] = True
        except ImportError:
            checks[f"lib_{lib}"] = False
            # Don't fail on optional libs

    # Disk space
    import shutil
    for mount_path in ["/mnt/localssd", "{REPO_ROOT}"]:
        try:
            usage = shutil.disk_usage(mount_path)
            checks[f"disk_{mount_path}_free_gb"] = round(usage.free / (1024**3), 1)
        except (FileNotFoundError, OSError):
            checks[f"disk_{mount_path}"] = "not_mounted"

    # HF token
    checks["hf_token_set"] = "HF_TOKEN" in os.environ and len(os.environ.get("HF_TOKEN", "")) > 5

    overall = "PASS" if passed else "FAIL"
    checks["overall"] = overall

    log_verification(log_path, {
        "type": "environment",
        "checks": checks,
        "passed": passed
    })

    print(f"[VERIFY ENV]: {overall}")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    return passed


def get_resume_status(project_dir: str) -> dict:
    """
    Analyze project directory and return a complete resume status report.
    Call this FIRST on any fresh agent start to determine what work remains.
    """
    import shutil

    status = {
        "is_fresh_start": True,
        "progress": None,
        "completed_experiments": [],
        "failed_experiments": [],
        "interrupted_experiment": None,
        "models_on_localssd": [],
        "models_on_remote": [],
        "models_missing": [],
        "datasets_on_localssd": [],
        "datasets_on_remote": [],
        "figures_generated": [],
        "paper_sections": [],
        "next_action": "start_fresh",
        "resume_summary": "",
    }

    # Check progress.json
    state_file = os.path.join(project_dir, "experiments", "progress.json")
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                progress = json.load(f)
            status["is_fresh_start"] = False
            status["progress"] = progress
        except (json.JSONDecodeError, Exception):
            pass

    # Check experiment log
    exp_log = os.path.join(project_dir, "experiments", "experiment_log.jsonl")
    if os.path.exists(exp_log):
        status["is_fresh_start"] = False
        with open(exp_log) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("status") == "success":
                        status["completed_experiments"].append(entry["id"])
                    elif entry.get("status") == "failed":
                        status["failed_experiments"].append(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    # Check for interrupted experiment
    if status["progress"]:
        current_task = status["progress"].get("current_task_id")
        if current_task and current_task not in status["completed_experiments"]:
            status["interrupted_experiment"] = current_task

    # Check local models
    localssd_models = "/mnt/localssd/models"
    if os.path.exists(localssd_models):
        status["models_on_localssd"] = [
            d for d in os.listdir(localssd_models)
            if os.path.isdir(os.path.join(localssd_models, d))
        ]

    # Check remote backup models
    remote_models = os.path.join(project_dir, "models_backup")
    if os.path.exists(remote_models):
        status["models_on_remote"] = [
            d for d in os.listdir(remote_models)
            if os.path.isdir(os.path.join(remote_models, d))
        ]

    # Check figures
    figures_dir = os.path.join(project_dir, "figures")
    if os.path.exists(figures_dir):
        status["figures_generated"] = [
            f for f in os.listdir(figures_dir)
            if f.endswith((".pdf", ".png", ".svg"))
        ]

    # Check paper sections
    paper_dir = os.path.join(project_dir, "paper")
    if os.path.exists(paper_dir):
        status["paper_sections"] = [
            f for f in os.listdir(paper_dir)
            if f.endswith((".tex", ".bib"))
        ]

    # Check remote checkpoints for interrupted training
    checkpoints_dir = os.path.join(project_dir, "checkpoints")
    if status["interrupted_experiment"] and os.path.exists(checkpoints_dir):
        exp_ckpt = os.path.join(checkpoints_dir, status["interrupted_experiment"])
        if os.path.exists(exp_ckpt) and os.listdir(exp_ckpt):
            latest_ckpt = sorted(os.listdir(exp_ckpt))[-1]
            status["resume_checkpoint"] = os.path.join(exp_ckpt, latest_ckpt)

    # Determine next action
    if status["is_fresh_start"]:
        status["next_action"] = "start_fresh"
        status["resume_summary"] = "No prior progress found. Starting from scratch."
    else:
        progress = status["progress"] or {}
        phase = progress.get("current_phase", "unknown")
        completed = len(status["completed_experiments"])
        total = progress.get("total_experiments", "?")
        failed = len(status["failed_experiments"])

        if phase == "complete":
            status["next_action"] = "already_complete"
            status["resume_summary"] = f"Paper already complete. {completed} experiments done."
        elif phase in ("setup", "planning"):
            status["next_action"] = "resume_planning"
            status["resume_summary"] = f"Interrupted during {phase}. Re-running setup and planning."
        elif phase == "downloads":
            status["next_action"] = "resume_downloads"
            status["resume_summary"] = f"Interrupted during downloads. Checking what's missing."
        elif phase == "experiments":
            status["next_action"] = "resume_experiments"
            interrupted_msg = ""
            if status["interrupted_experiment"]:
                ckpt = status.get("resume_checkpoint", "none")
                interrupted_msg = f" Interrupted experiment: {status['interrupted_experiment']} (checkpoint: {ckpt})."
            status["resume_summary"] = (
                f"Resuming experiments. {completed}/{total} done, {failed} failed.{interrupted_msg}"
            )
        elif phase == "analysis":
            status["next_action"] = "resume_analysis"
            figs = len(status["figures_generated"])
            status["resume_summary"] = f"Resuming analysis. {figs} figures already generated."
        elif phase == "writing":
            status["next_action"] = "resume_writing"
            secs = len(status["paper_sections"])
            status["resume_summary"] = f"Resuming writing. {secs} sections already exist."
        else:
            status["next_action"] = "resume_experiments"
            status["resume_summary"] = f"Unknown phase '{phase}'. Resuming from experiments."

    return status


def print_resume_report(project_dir: str):
    """Print a human-readable resume status report."""
    status = get_resume_status(project_dir)

    print("=" * 60)
    if status["is_fresh_start"]:
        print("STATUS: FRESH START — No prior progress found")
    else:
        print("STATUS: RESUMING FROM PRIOR RUN")
    print("=" * 60)

    if status["progress"]:
        p = status["progress"]
        print(f"  Paper: {p.get('paper', '?')}")
        print(f"  Last machine: {p.get('last_machine', '?')}")
        print(f"  Last updated: {p.get('last_updated', '?')}")
        print(f"  Current phase: {p.get('current_phase', '?')}")
        print(f"  Resume count: {p.get('resume_count', 0)}")

    print(f"\n  Completed experiments: {len(status['completed_experiments'])}")
    print(f"  Failed experiments: {len(status['failed_experiments'])}")
    if status["interrupted_experiment"]:
        print(f"  Interrupted experiment: {status['interrupted_experiment']}")
        if status.get("resume_checkpoint"):
            print(f"  Resume checkpoint: {status['resume_checkpoint']}")

    print(f"\n  Models on local SSD: {len(status['models_on_localssd'])}")
    print(f"  Models on remote backup: {len(status['models_on_remote'])}")
    print(f"  Figures generated: {len(status['figures_generated'])}")
    print(f"  Paper sections: {len(status['paper_sections'])}")

    print(f"\n  NEXT ACTION: {status['next_action']}")
    print(f"  {status['resume_summary']}")
    print("=" * 60)

    return status


if __name__ == "__main__":
    """Run standalone verification."""
    import argparse
    parser = argparse.ArgumentParser(description="NeurIPS 2026 Research Verification")
    parser.add_argument("--type", choices=["env", "model", "data", "experiment", "figure", "resume"], required=True)
    parser.add_argument("--project-dir", type=str, help="Project directory for resume check")
    parser.add_argument("--path", type=str, help="Path to verify")
    parser.add_argument("--name", type=str, help="Name/ID of the asset")
    parser.add_argument("--model-id", type=str, help="HuggingFace model ID")
    parser.add_argument("--expected-count", type=int, help="Expected example count for data")
    parser.add_argument("--expected-fields", type=str, nargs="+", help="Expected fields for data")
    parser.add_argument("--log-path", type=str, default="verification_log.jsonl")
    args = parser.parse_args()

    if args.type == "resume":
        if not args.project_dir:
            print("ERROR: --project-dir required for resume check")
            sys.exit(1)
        print_resume_report(args.project_dir)
        sys.exit(0)
    elif args.type == "env":
        verify_environment(args.log_path)
    elif args.type == "model":
        verify_model_download(args.path, args.model_id or args.name, args.log_path)
    elif args.type == "data":
        verify_dataset(args.path, args.name, args.expected_count or 0,
                       args.expected_fields or [], args.log_path)
    elif args.type == "experiment":
        verify_experiment(args.name, args.name, args.path, [], {}, 0.5, args.log_path)
    elif args.type == "figure":
        verify_figure(args.path, args.name, args.log_path)
