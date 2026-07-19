# Pipeline 04: bentuk protokol eksperimen WIDER FACE.
"""
Build the final WIDER experiment protocol using lightweight manifest files.

Protocol:
- Source train split  : official WIDER train converted to YOLO
- Source val split    : official WIDER val converted to YOLO
- Final train split   : 90% of source train (`train_fit`)
- Final val split     : 10% of source train (`train_dev`) for model selection
- Final test split    : official WIDER val for held-out local evaluation

Output layout:
  datasets/experiment/wider_face_clean/
    train.txt
    val.txt
    test.txt
    PROTOCOL_INFO.txt
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from project_config import (
    EXPECTED_SPLITS,
    IMAGE_EXTENSIONS,
    WIDER_CLEAN_DIR,
    WIDER_PROTOCOL_DEV_RATIO,
    WIDER_PROTOCOL_SEED,
    WIDER_YOLO_RAW_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare final WIDER experiment manifests from the YOLO-converted official dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=WIDER_YOLO_RAW_DIR,
        help="Root dataset WIDER yang sudah dikonversi ke format YOLO.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=WIDER_CLEAN_DIR,
        help="Root target dataset clean protokol final.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=WIDER_PROTOCOL_SEED,
        help="Seed untuk split train_fit/train_dev.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=WIDER_PROTOCOL_DEV_RATIO,
        help="Proporsi source train yang dipakai sebagai train_dev.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Hapus target root sebelum rebuild.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} tidak ditemukan: {path}")


def list_split_images(images_dir: Path, labels_dir: Path) -> list[Path]:
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_stems = {
        path.stem for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    }
    image_stems = {path.stem for path in image_paths}
    if image_stems != label_stems:
        missing = sorted(image_stems - label_stems)
        orphan = sorted(label_stems - image_stems)
        raise ValueError(
            f"Source split tidak konsisten. missing_labels={len(missing)}, orphan_labels={len(orphan)}"
        )
    return image_paths


def write_manifest(target_root: Path, name: str, image_paths: list[Path]) -> None:
    manifest_path = target_root / f"{name}.txt"
    lines = [str(path.resolve()) for path in image_paths]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_note(
    target_root: Path,
    train_paths: list[Path],
    dev_paths: list[Path],
    test_paths: list[Path],
    seed: int,
    dev_ratio: float,
) -> None:
    note_path = target_root / "PROTOCOL_INFO.txt"
    lines = [
        "source=official wider_face converted to yolo",
        "storage=manifest_only",
        f"seed={seed}",
        f"dev_ratio={dev_ratio}",
        f"train_count={len(train_paths)}",
        f"val_count={len(dev_paths)}",
        f"test_count={len(test_paths)}",
        "",
        "notes:",
        "- Final train split berasal dari source train resmi WIDER.",
        "- Final val split adalah train_dev untuk model selection.",
        "- Final test split adalah official WIDER validation set untuk held-out local evaluation.",
        "- Official WIDER test set tidak dipakai untuk mAP lokal.",
    ]
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_protocol_dataset(
    source_root: Path,
    target_root: Path,
    seed: int,
    dev_ratio: float,
    clear_target: bool,
) -> None:
    if clear_target and target_root.exists():
        shutil.rmtree(target_root, ignore_errors=True)
    target_root.mkdir(parents=True, exist_ok=True)

    train_img_dir = source_root / "images" / "train"
    train_lbl_dir = source_root / "labels" / "train"
    test_img_dir = source_root / "images" / "val"
    test_lbl_dir = source_root / "labels" / "val"

    ensure_exists(train_img_dir, "Source train images")
    ensure_exists(train_lbl_dir, "Source train labels")
    ensure_exists(test_img_dir, "Source val images")
    ensure_exists(test_lbl_dir, "Source val labels")

    source_train_paths = list_split_images(train_img_dir, train_lbl_dir)
    source_test_paths = list_split_images(test_img_dir, test_lbl_dir)

    rng = random.Random(seed)
    shuffled = list(source_train_paths)
    rng.shuffle(shuffled)

    dev_count = int(len(shuffled) * dev_ratio)
    if dev_count <= 0 or dev_count >= len(shuffled):
        raise ValueError(f"dev_ratio tidak valid untuk jumlah source train: {dev_ratio}")

    dev_paths = sorted(shuffled[:dev_count], key=lambda path: path.name)
    train_paths = sorted(shuffled[dev_count:], key=lambda path: path.name)
    test_paths = source_test_paths

    expected = EXPECTED_SPLITS["wider_face"]
    actual = {
        "train": len(train_paths),
        "val": len(dev_paths),
        "test": len(test_paths),
    }
    if actual != expected:
        raise ValueError(f"Protocol split count mismatch. expected={expected}, actual={actual}")

    write_manifest(target_root, "train", train_paths)
    write_manifest(target_root, "val", dev_paths)
    write_manifest(target_root, "test", test_paths)
    write_note(target_root, train_paths, dev_paths, test_paths, seed, dev_ratio)

    print("WIDER protocol manifest ready:")
    print(f"  train -> {len(train_paths)}")
    print(f"  val   -> {len(dev_paths)}")
    print(f"  test  -> {len(test_paths)}")


def main() -> None:
    args = parse_args()
    build_protocol_dataset(
        source_root=args.source_root.resolve(),
        target_root=args.target_root.resolve(),
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        clear_target=args.clear,
    )


if __name__ == "__main__":
    main()
