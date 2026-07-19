# Pipeline 12: evaluasi robustness seluruh skenario.
"""
Evaluate leakage-free robustness on WIDER test, synthetic WIDER test_lowlight,
and optional Dark Face test.
"""
from __future__ import annotations

import csv
from pathlib import Path
import tempfile

import yaml
from ultralytics import YOLO

from scripts.audit_dataset_09 import run_full_audit
from project_config import (
    AUGMENTED_WEIGHTS,
    BASELINE_WEIGHTS,
    DARK_CLEAN_DIR,
    EVAL_RESULTS_DIR,
    WIDER_AUG_DIR,
    WIDER_CLEAN_DIR,
)


def split_entry(dataset_root: Path, split_name: str) -> str:
    manifest = dataset_root / f"{split_name}.txt"
    if manifest.exists():
        return manifest.name

    image_dir = dataset_root / "images" / split_name
    if image_dir.exists():
        return f"images/{split_name}"

    raise FileNotFoundError(f"Split entry tidak ditemukan: {dataset_root}:{split_name}")


def create_eval_yaml(dataset_root: Path, split_name: str) -> Path:
    try:
        train_entry = split_entry(dataset_root, "train")
    except FileNotFoundError:
        train_entry = split_entry(dataset_root, split_name)

    data = {
        "path": str(dataset_root.resolve()),
        "train": train_entry,
        "val": split_entry(dataset_root, split_name),
        "names": {0: "face"},
    }

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{dataset_root.name}_{split_name}.yaml",
        delete=False,
        encoding="utf-8",
    )
    yaml.dump(data, handle, default_flow_style=False)
    handle.close()
    return Path(handle.name)


def run_evaluation(weights_path: Path, dataset_root: Path, split_name: str, test_name: str):
    print(
        f"\nEvaluating: {test_name} | "
        f"weights={weights_path.parent.parent.name} | dataset={dataset_root.name}:{split_name}"
    )
    yaml_path = create_eval_yaml(dataset_root, split_name)
    try:
        model = YOLO(str(weights_path))
        results = model.val(
            data=str(yaml_path),
            imgsz=640,
            batch=16,
            device=0,
            plots=True,
            save_json=False,
            project=str(EVAL_RESULTS_DIR),
            name=test_name,
        )
    finally:
        yaml_path.unlink(missing_ok=True)

    metrics = {
        "test_name": test_name,
        "precision": round(results.box.mp, 5),
        "recall": round(results.box.mr, 5),
        "mAP50": round(results.box.map50, 5),
        "mAP50-95": round(results.box.map, 5),
    }
    print(
        "Results "
        + test_name
        + " - "
        + " | ".join(f"{key}: {value}" for key, value in metrics.items() if key != "test_name")
    )
    return metrics


def print_comparison_table(all_results):
    print("\n[Comparison Summary]")
    for result in all_results:
        print(
            f"{result['test_name']:<24}: "
            f"mAP50={result['mAP50']:.4f}, mAP50-95={result['mAP50-95']:.4f}"
        )

    baseline_normal = next(r for r in all_results if r["test_name"] == "baseline_normal")
    baseline_lowlight = next(r for r in all_results if r["test_name"] == "baseline_lowlight")
    augmented_normal = next(r for r in all_results if r["test_name"] == "augmented_normal")
    augmented_lowlight = next(r for r in all_results if r["test_name"] == "augmented_lowlight")

    drop_baseline = (
        (baseline_normal["mAP50"] - baseline_lowlight["mAP50"])
        / baseline_normal["mAP50"]
        * 100
    )
    drop_augmented = (
        (augmented_normal["mAP50"] - augmented_lowlight["mAP50"])
        / augmented_normal["mAP50"]
        * 100
    )

    print(f"Robustness Drop | Baseline: {drop_baseline:.2f}% | Augmented: {drop_augmented:.2f}%")
    return {
        "drop_baseline": drop_baseline,
        "drop_augmented": drop_augmented,
    }


def save_results_csv(all_results):
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_RESULTS_DIR / "robustness_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["test_name", "precision", "recall", "mAP50", "mAP50-95"],
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved to: {csv_path}")


def main():
    require_darkface = DARK_CLEAN_DIR.exists()
    run_full_audit(require_darkface=require_darkface)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, path in (("Baseline", BASELINE_WEIGHTS), ("Augmented", AUGMENTED_WEIGHTS)):
        if not path.exists():
            raise FileNotFoundError(f"{name} weights not found at {path}")
        print(f"{name} weights: {path}")

    all_results = [
        run_evaluation(BASELINE_WEIGHTS, WIDER_CLEAN_DIR, "test", "baseline_normal"),
        run_evaluation(BASELINE_WEIGHTS, WIDER_CLEAN_DIR, "test_lowlight", "baseline_lowlight"),
        run_evaluation(AUGMENTED_WEIGHTS, WIDER_AUG_DIR, "test", "augmented_normal"),
        run_evaluation(AUGMENTED_WEIGHTS, WIDER_AUG_DIR, "test_lowlight", "augmented_lowlight"),
    ]

    if require_darkface:
        all_results.append(
            run_evaluation(BASELINE_WEIGHTS, DARK_CLEAN_DIR, "test", "baseline_darkface")
        )
        all_results.append(
            run_evaluation(AUGMENTED_WEIGHTS, DARK_CLEAN_DIR, "test", "augmented_darkface")
        )

    print_comparison_table(all_results)
    save_results_csv(all_results)
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
