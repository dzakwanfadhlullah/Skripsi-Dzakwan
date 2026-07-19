# Pipeline 08: siapkan Dark Face sebagai external test.
"""
Prepare Dark Face as an external low-light evaluation dataset.

Input:
  datasets/raw/dark_face/
    image/*.png
    label/*.txt

Raw label format:
  line 1  : number of boxes
  line 2+ : x1 y1 x2 y2

Output:
  datasets/experiment/dark_face_clean/
    images/test/*
    labels/test/*
    test.txt
    CONVERSION_INFO.txt
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image

from project_config import DARK_CLEAN_DIR, DARK_RAW_DIR, IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Dark Face raw labels into a YOLO-compatible external test dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DARK_RAW_DIR,
        help="Root raw Dark Face dataset.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DARK_CLEAN_DIR,
        help="Target clean dataset root for external evaluation.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete target root before rebuilding it.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} tidak ditemukan: {path}")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.unlink(missing_ok=True)
        os.link(source, destination)
    except Exception:
        shutil.copy2(source, destination)


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def convert_label(raw_label_path: Path, image_width: int, image_height: int) -> tuple[list[str], dict[str, int]]:
    lines = [line.strip() for line in raw_label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Label kosong: {raw_label_path}")

    declared = int(lines[0])
    boxes = lines[1:]
    if declared != len(boxes):
        raise ValueError(
            f"Header label tidak cocok di {raw_label_path.name}: declared={declared}, actual={len(boxes)}"
        )

    converted: list[str] = []
    stats = {
        "boxes_total": 0,
        "boxes_clipped": 0,
        "boxes_skipped": 0,
    }

    for line in boxes:
        x1, y1, x2, y2 = map(float, line.split())
        stats["boxes_total"] += 1

        clipped_x1 = min(max(x1, 0.0), float(image_width))
        clipped_y1 = min(max(y1, 0.0), float(image_height))
        clipped_x2 = min(max(x2, 0.0), float(image_width))
        clipped_y2 = min(max(y2, 0.0), float(image_height))

        if (clipped_x1, clipped_y1, clipped_x2, clipped_y2) != (x1, y1, x2, y2):
            stats["boxes_clipped"] += 1

        box_w = clipped_x2 - clipped_x1
        box_h = clipped_y2 - clipped_y1
        if box_w <= 0 or box_h <= 0:
            stats["boxes_skipped"] += 1
            continue

        x_center = (clipped_x1 + clipped_x2) / 2.0 / image_width
        y_center = (clipped_y1 + clipped_y2) / 2.0 / image_height
        norm_w = box_w / image_width
        norm_h = box_h / image_height
        converted.append(
            f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        )

    return converted, stats


def write_manifest(target_root: Path, image_paths: list[Path]) -> None:
    manifest_path = target_root / "test.txt"
    manifest_path.write_text(
        "\n".join(str(path.resolve()) for path in image_paths) + "\n",
        encoding="utf-8",
    )


def write_note(target_root: Path, summary: dict[str, int]) -> None:
    lines = [
        "source=dark_face_raw",
        "usage=external_test_only",
        f"image_count={summary['image_count']}",
        f"label_count={summary['label_count']}",
        f"boxes_total={summary['boxes_total']}",
        f"boxes_written={summary['boxes_written']}",
        f"boxes_clipped={summary['boxes_clipped']}",
        f"boxes_skipped={summary['boxes_skipped']}",
        "",
        "notes:",
        "- Dataset ini dipakai hanya sebagai evaluasi eksternal low-light.",
        "- Dark Face tidak dipakai untuk training atau model selection.",
    ]
    (target_root / "CONVERSION_INFO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_darkface_external(source_root: Path, target_root: Path, clear_target: bool) -> None:
    if clear_target and target_root.exists():
        shutil.rmtree(target_root, ignore_errors=True)

    image_dir = source_root / "image"
    label_dir = source_root / "label"
    ensure_exists(image_dir, "Dark Face image directory")
    ensure_exists(label_dir, "Dark Face label directory")

    target_img_dir = target_root / "images" / "test"
    target_lbl_dir = target_root / "labels" / "test"
    shutil.rmtree(target_img_dir, ignore_errors=True)
    shutil.rmtree(target_lbl_dir, ignore_errors=True)
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(image_dir)
    written_images: list[Path] = []
    summary = {
        "image_count": 0,
        "label_count": 0,
        "boxes_total": 0,
        "boxes_written": 0,
        "boxes_clipped": 0,
        "boxes_skipped": 0,
    }

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        ensure_exists(label_path, f"Dark Face label {image_path.stem}")
        summary["image_count"] += 1
        summary["label_count"] += 1

        with Image.open(image_path) as image:
            width, height = image.size

        converted_lines, stats = convert_label(label_path, width, height)
        summary["boxes_total"] += stats["boxes_total"]
        summary["boxes_clipped"] += stats["boxes_clipped"]
        summary["boxes_skipped"] += stats["boxes_skipped"]
        summary["boxes_written"] += len(converted_lines)

        target_image_path = target_img_dir / image_path.name
        target_label_path = target_lbl_dir / f"{image_path.stem}.txt"
        link_or_copy(image_path, target_image_path)
        target_label_path.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")
        written_images.append(target_image_path)

    write_manifest(target_root, written_images)
    write_note(target_root, summary)

    print("Dark Face external dataset ready:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    prepare_darkface_external(
        source_root=args.source_root.resolve(),
        target_root=args.target_root.resolve(),
        clear_target=args.clear,
    )


if __name__ == "__main__":
    main()
