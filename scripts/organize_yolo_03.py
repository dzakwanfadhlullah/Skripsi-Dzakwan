# Pipeline 03: susun dataset YOLO berdasarkan split resmi.
"""
Prepare clean YOLO datasets using official split counts.

Expected raw layouts:
1. YOLO-separated:
   datasets/raw/<dataset_name>/images/<split>/*
   datasets/raw/<dataset_name>/labels/<split>/*

2. YOLO-mixed:
   datasets/raw/<dataset_name>/<split>/*
   where image files and label txt live in the same folder.

3. Flat YOLO folders plus manifest:
   datasets/raw/<dataset_name>/images/*
   datasets/raw/<dataset_name>/labels/*
   datasets/raw/<dataset_name>/splits/train.txt
   datasets/raw/<dataset_name>/splits/val.txt
   datasets/raw/<dataset_name>/splits/test.txt

This script copies raw official splits into immutable clean datasets under
`datasets/experiment/` and validates the expected image counts.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from project_config import (
    DARK_CLEAN_DIR,
    DARK_RAW_DIR,
    EXPECTED_SPLITS,
    IMAGE_EXTENSIONS,
    WIDER_CLEAN_DIR,
    WIDER_RAW_DIR,
)


IGNORED_LABEL_NAMES = {"classes", "notes", "readme"}
DATASET_TARGETS = {
    "wider_face": (WIDER_RAW_DIR, WIDER_CLEAN_DIR),
    "dark_face": (DARK_RAW_DIR, DARK_CLEAN_DIR),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy official raw YOLO datasets into clean experiment folders."
    )
    parser.add_argument(
        "--dataset",
        choices=("wider_face", "dark_face", "all"),
        default="all",
        help="Dataset to prepare.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the target clean dataset before rebuilding it.",
    )
    return parser.parse_args()


def resolve_split_dirs(source_root: Path, split_name: str) -> tuple[Path, Path]:
    image_candidates = [
        source_root / "images" / split_name,
        source_root / split_name,
    ]
    label_candidates = [
        source_root / "labels" / split_name,
        source_root / split_name,
    ]

    image_dir = next((path for path in image_candidates if path.is_dir()), None)
    label_dir = next((path for path in label_candidates if path.is_dir()), None)

    if image_dir is None:
        raise FileNotFoundError(
            f"Split '{split_name}' tidak ditemukan di bawah {source_root}."
        )
    if label_dir is None:
        raise FileNotFoundError(
            f"Folder label untuk split '{split_name}' tidak ditemukan di bawah {source_root}."
        )

    return image_dir, label_dir


def split_manifest_path(source_root: Path, split_name: str) -> Path:
    return source_root / "splits" / f"{split_name}.txt"


def list_images(image_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def list_labels(label_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in label_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".txt"
        and path.stem.lower() not in IGNORED_LABEL_NAMES
    }


def normalize_manifest_entry(entry: str) -> str:
    cleaned = entry.strip()
    if not cleaned or cleaned.startswith("#"):
        return ""
    return Path(cleaned).stem


def load_manifest_stems(manifest_path: Path) -> list[str]:
    stems = [
        normalize_manifest_entry(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    return [stem for stem in stems if stem]


def load_split_pairs(
    dataset_name: str,
    source_root: Path,
    split_name: str,
) -> tuple[dict[str, Path], dict[str, Path]]:
    try:
        image_dir, label_dir = resolve_split_dirs(source_root, split_name)
    except FileNotFoundError:
        image_dir = source_root / "images"
        label_dir = source_root / "labels"
        manifest_path = split_manifest_path(source_root, split_name)

        if not image_dir.is_dir() or not label_dir.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Split '{split_name}' untuk {dataset_name} tidak ditemukan. "
                "Gunakan salah satu layout yang didukung: split directories, mixed split "
                "directories, atau folder flat images/labels dengan manifest splits/<split>.txt."
            )

        all_images = list_images(image_dir)
        all_labels = list_labels(label_dir)
        stems = load_manifest_stems(manifest_path)

        images = {stem: all_images[stem] for stem in stems if stem in all_images}
        labels = {stem: all_labels[stem] for stem in stems if stem in all_labels}
        validate_pairs(dataset_name, split_name, images, labels)

        if len(images) != len(stems):
            missing_images = sorted(set(stems) - set(images))
            raise ValueError(
                f"{dataset_name}:{split_name} manifest tidak cocok dengan folder images. "
                f"Missing images={len(missing_images)}."
            )

        if len(labels) != len(stems):
            missing_labels = sorted(set(stems) - set(labels))
            raise ValueError(
                f"{dataset_name}:{split_name} manifest tidak cocok dengan folder labels. "
                f"Missing labels={len(missing_labels)}."
            )

        return images, labels

    images = list_images(image_dir)
    labels = list_labels(label_dir)
    validate_pairs(dataset_name, split_name, images, labels)
    return images, labels


def copy_pairs(
    images: dict[str, Path],
    labels: dict[str, Path],
    target_root: Path,
    split_name: str,
) -> int:
    target_img_dir = target_root / "images" / split_name
    target_lbl_dir = target_root / "labels" / split_name
    shutil.rmtree(target_img_dir, ignore_errors=True)
    shutil.rmtree(target_lbl_dir, ignore_errors=True)
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    for stem in sorted(images):
        image_path = images[stem]
        label_path = labels[stem]
        shutil.copy2(image_path, target_img_dir / image_path.name)
        shutil.copy2(label_path, target_lbl_dir / f"{stem}.txt")

    return len(images)


def validate_pairs(
    dataset_name: str,
    split_name: str,
    images: dict[str, Path],
    labels: dict[str, Path],
) -> None:
    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    if missing_labels or orphan_labels:
        raise ValueError(
            f"{dataset_name}:{split_name} tidak konsisten. "
            f"Missing labels={len(missing_labels)}, orphan labels={len(orphan_labels)}."
        )


def prepare_dataset(
    dataset_name: str,
    source_root: Path,
    target_root: Path,
    clear_target: bool = False,
) -> None:
    expected_splits = EXPECTED_SPLITS[dataset_name]

    if clear_target and target_root.exists():
        shutil.rmtree(target_root)

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source dataset {dataset_name} tidak ditemukan: {source_root}"
        )

    print(f"\n=== Preparing {dataset_name} ===")
    print(f"Source : {source_root}")
    print(f"Target : {target_root}")

    copied_total = 0
    for split_name, expected_count in expected_splits.items():
        images, labels = load_split_pairs(dataset_name, source_root, split_name)

        if len(images) != expected_count:
            raise ValueError(
                f"{dataset_name}:{split_name} count mismatch. "
                f"Expected {expected_count}, found {len(images)}. "
                f"Pastikan Anda memakai dataset resmi dengan split official."
            )

        copied = copy_pairs(images, labels, target_root, split_name)
        copied_total += copied
        print(f"  {split_name:<5} -> {copied} pairs")

    print(f"Selesai menyalin {copied_total} pasangan untuk {dataset_name}.")


def main() -> None:
    args = parse_args()
    dataset_names = list(DATASET_TARGETS) if args.dataset == "all" else [args.dataset]

    for dataset_name in dataset_names:
        source_root, target_root = DATASET_TARGETS[dataset_name]
        prepare_dataset(dataset_name, source_root, target_root, clear_target=args.clear)

    print(
        "\nDataset clean siap. Lanjutkan dengan generate low-light test untuk WIDER, "
        "build augmented train, lalu jalankan audit anti-leak."
    )


if __name__ == "__main__":
    main()
