# Pipeline 07: bentuk dataset training augmented.
"""
Build a leakage-free augmented WIDER dataset.

Input  : clean WIDER dataset with manifest-based train/val/test splits
Output : augmented WIDER dataset where only the train split is materialized
         and augmented, while val/test manifests are reused.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from scripts.augment_data_helper_05 import augment_file_to_low_light
from project_config import IMAGE_EXTENSIONS, WIDER_AUG_DIR, WIDER_CLEAN_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a clean augmented WIDER dataset from the clean baseline dataset."
    )
    parser.add_argument(
        "--source-root",
        default=str(WIDER_CLEAN_DIR),
        help="Path ke root dataset clean WIDER.",
    )
    parser.add_argument(
        "--target-root",
        default=str(WIDER_AUG_DIR),
        help="Path ke root dataset augmented WIDER.",
    )
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--brightness", type=int, default=-30)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Hapus target augmented sebelum rebuild.",
    )
    return parser.parse_args()


def resolve_manifest_entry(dataset_root: Path, raw_line: str) -> Path:
    path = Path(raw_line.strip())
    if not path.is_absolute():
        path = (dataset_root / path).resolve()
    return path


def load_split_images(dataset_root: Path, split_name: str) -> list[Path]:
    image_dir = dataset_root / "images" / split_name
    if image_dir.exists():
        return sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    manifest_path = dataset_root / f"{split_name}.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split {split_name} tidak ditemukan di {dataset_root}")

    lines = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [resolve_manifest_entry(dataset_root, line) for line in lines]


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise FileNotFoundError(f"Tidak bisa menginfer label dari image path: {image_path}")


def write_manifest(target_root: Path, split_name: str, image_paths: list[Path]) -> None:
    manifest_path = target_root / f"{split_name}.txt"
    manifest_path.write_text(
        "\n".join(str(path.resolve()) for path in image_paths) + "\n",
        encoding="utf-8",
    )


def validate_clean_train_source(image_paths: list[Path]) -> None:
    invalid = [path.name for path in image_paths if path.name.startswith("low_")]
    if invalid:
        raise ValueError(
            "Source train split sudah terkontaminasi file low-light. "
            f"Contoh: {invalid[:5]}"
        )


def build_augmented_train(
    source_root: Path,
    target_root: Path,
    gamma: float,
    brightness: int,
) -> None:
    dst_img_dir = target_root / "images" / "train"
    dst_lbl_dir = target_root / "labels" / "train"
    shutil.rmtree(dst_img_dir, ignore_errors=True)
    shutil.rmtree(dst_lbl_dir, ignore_errors=True)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_split_images(source_root, "train")
    validate_clean_train_source(image_paths)
    print(f"Building augmented train split from {len(image_paths)} clean images...")

    written_paths: list[Path] = [path.resolve() for path in image_paths]
    for image_path in tqdm(image_paths, desc="Augmenting train"):
        label_path = infer_label_path(image_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Label tidak ditemukan untuk {image_path.name}")

        low_output = dst_img_dir / f"low_{image_path.name}"
        low_label_output = dst_lbl_dir / f"low_{label_path.name}"

        ok = augment_file_to_low_light(
            image_path,
            low_output,
            gamma=gamma,
            brightness=brightness,
        )
        if not ok:
            raise RuntimeError(f"Gagal augment gambar {image_path}")
        shutil.copy2(label_path, low_label_output)

        written_paths.append(low_output.resolve())

    write_manifest(target_root, "train", written_paths)


def reuse_eval_split_manifests(source_root: Path, target_root: Path) -> None:
    for split_name in ("val", "test", "test_lowlight"):
        source_manifest = source_root / f"{split_name}.txt"
        if source_manifest.exists():
            shutil.copy2(source_manifest, target_root / source_manifest.name)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()

    if args.clear and target_root.exists():
        shutil.rmtree(target_root, ignore_errors=True)
    target_root.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise FileNotFoundError(f"Dataset clean tidak ditemukan: {source_root}")

    build_augmented_train(
        source_root=source_root,
        target_root=target_root,
        gamma=args.gamma,
        brightness=args.brightness,
    )
    reuse_eval_split_manifests(source_root, target_root)

    print(f"Augmented dataset ready at: {target_root}")


if __name__ == "__main__":
    main()
