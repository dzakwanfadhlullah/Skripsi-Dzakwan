"""
Convert official WIDER FACE annotations into a YOLO-style dataset.

Output layout:
  datasets/raw/wider_face_yolo/
    images/train
    images/val
    images/test
    labels/train
    labels/val
    splits/train.txt
    splits/val.txt
    splits/test.txt

Important:
- Train and val receive YOLO labels converted from the official
  `wider_face_*_bbx_gt.txt` files.
- Test only receives images and a manifest because WIDER FACE does not publish
  test bounding boxes for local evaluation.
"""
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from project_config import WIDER_RAW_DIR, WIDER_YOLO_RAW_DIR


@dataclass
class ConversionStats:
    images: int = 0
    boxes_written: int = 0
    boxes_skipped: int = 0
    empty_labels: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official WIDER FACE train/val annotations to YOLO format."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=WIDER_RAW_DIR,
        help="Root of the official WIDER FACE raw dataset.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=WIDER_YOLO_RAW_DIR,
        help="Target root for the converted YOLO-style dataset.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the target folder before conversion.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} tidak ditemukan: {path}")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def to_yolo_line(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> str | None:
    xmin = clamp(x, 0, image_w)
    ymin = clamp(y, 0, image_h)
    xmax = clamp(x + w, 0, image_w)
    ymax = clamp(y + h, 0, image_h)

    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2.0) / image_w
    y_center = ((ymin + ymax) / 2.0) / image_h
    width = (xmax - xmin) / image_w
    height = (ymax - ymin) / image_h

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def convert_split(
    source_root: Path,
    target_root: Path,
    split_name: str,
    annotation_file: Path,
) -> ConversionStats:
    image_source_root = source_root / f"WIDER_{split_name}" / "images"
    image_target_root = target_root / "images" / split_name
    label_target_root = target_root / "labels" / split_name
    manifest_path = target_root / "splits" / f"{split_name}.txt"

    ensure_exists(image_source_root, f"WIDER_{split_name} images")
    ensure_exists(annotation_file, f"WIDER {split_name} annotation file")

    image_target_root.mkdir(parents=True, exist_ok=True)
    label_target_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats()
    manifest_entries: list[str] = []

    lines = annotation_file.read_text(encoding="utf-8").splitlines()
    index = 0
    total_lines = len(lines)

    while index < total_lines:
        relative_path = lines[index].strip()
        index += 1
        if not relative_path:
            continue

        if index >= total_lines:
            raise ValueError(f"Format anotasi {annotation_file} terpotong di akhir file.")

        box_count = int(lines[index].strip())
        index += 1

        image_path = image_source_root / relative_path
        ensure_exists(image_path, f"Image {split_name}")
        image_w, image_h = load_image_size(image_path)

        destination_image = image_target_root / image_path.name
        destination_label = label_target_root / f"{image_path.stem}.txt"
        link_or_copy(image_path, destination_image)

        yolo_lines: list[str] = []
        if box_count == 0:
            if index >= total_lines:
                raise ValueError(
                    f"Format anotasi {annotation_file} terpotong setelah zero-box entry {relative_path}."
                )
            # Official WIDER annotations keep one dummy line after zero-box entries.
            index += 1

        for _ in range(box_count):
            if index >= total_lines:
                raise ValueError(
                    f"Format anotasi {annotation_file} terpotong saat membaca bbox {relative_path}."
                )
            parts = lines[index].split()
            index += 1
            if len(parts) < 4:
                raise ValueError(f"Baris bbox tidak valid untuk {relative_path}: {parts}")
            x, y, w, h = map(int, parts[:4])
            yolo_line = to_yolo_line(x, y, w, h, image_w, image_h)
            if yolo_line is None:
                stats.boxes_skipped += 1
                continue
            yolo_lines.append(yolo_line)
            stats.boxes_written += 1

        if not yolo_lines:
            stats.empty_labels += 1
            destination_label.write_text("", encoding="utf-8")
        else:
            destination_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        manifest_entries.append(destination_image.name)
        stats.images += 1

    manifest_path.write_text("\n".join(manifest_entries) + "\n", encoding="utf-8")
    return stats


def prepare_test_split(source_root: Path, target_root: Path, filelist_path: Path) -> int:
    image_source_root = source_root / "WIDER_test" / "images"
    image_target_root = target_root / "images" / "test"
    manifest_path = target_root / "splits" / "test.txt"
    relative_manifest_path = target_root / "splits" / "test_relative_paths.txt"

    ensure_exists(image_source_root, "WIDER_test images")
    ensure_exists(filelist_path, "WIDER test filelist")

    image_target_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    relative_paths = [line.strip() for line in filelist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    image_names: list[str] = []
    for relative_path in relative_paths:
        image_path = image_source_root / relative_path
        ensure_exists(image_path, "WIDER test image")
        destination_image = image_target_root / image_path.name
        link_or_copy(image_path, destination_image)
        image_names.append(destination_image.name)

    manifest_path.write_text("\n".join(image_names) + "\n", encoding="utf-8")
    relative_manifest_path.write_text("\n".join(relative_paths) + "\n", encoding="utf-8")
    return len(image_names)


def write_notes(target_root: Path, train_stats: ConversionStats, val_stats: ConversionStats, test_count: int) -> None:
    note_path = target_root / "CONVERSION_INFO.txt"
    lines = [
        "source=official wider_face benchmark",
        f"train_images={train_stats.images}",
        f"val_images={val_stats.images}",
        f"test_images={test_count}",
        f"train_boxes_written={train_stats.boxes_written}",
        f"val_boxes_written={val_stats.boxes_written}",
        f"train_boxes_skipped={train_stats.boxes_skipped}",
        f"val_boxes_skipped={val_stats.boxes_skipped}",
        f"train_empty_labels={train_stats.empty_labels}",
        f"val_empty_labels={val_stats.empty_labels}",
        "",
        "notes:",
        "- Train dan val sudah dikonversi ke label YOLO class 0 (face).",
        "- Split test hanya berisi image dan manifest karena ground truth resmi WIDER FACE test tidak dipublikasikan untuk evaluasi lokal.",
        "- Bounding box diklip ke ukuran gambar seperti implementasi builder TFDS resmi.",
    ]
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_counts(train_stats: ConversionStats, val_stats: ConversionStats, test_count: int) -> None:
    expected = {"train": 12880, "val": 3226, "test": 16097}
    actual = {"train": train_stats.images, "val": val_stats.images, "test": test_count}
    if actual != expected:
        raise ValueError(f"Count hasil konversi tidak cocok. expected={expected}, actual={actual}")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()

    if args.clear and target_root.exists():
        shutil.rmtree(target_root)

    ensure_exists(source_root, "WIDER official source root")
    ensure_exists(source_root / "wider_face_split", "WIDER wider_face_split")

    print(f"Source : {source_root}")
    print(f"Target : {target_root}")

    train_stats = convert_split(
        source_root,
        target_root,
        "train",
        source_root / "wider_face_split" / "wider_face_train_bbx_gt.txt",
    )
    print(
        f"train -> images={train_stats.images}, boxes={train_stats.boxes_written}, "
        f"skipped={train_stats.boxes_skipped}, empty={train_stats.empty_labels}"
    )

    val_stats = convert_split(
        source_root,
        target_root,
        "val",
        source_root / "wider_face_split" / "wider_face_val_bbx_gt.txt",
    )
    print(
        f"val   -> images={val_stats.images}, boxes={val_stats.boxes_written}, "
        f"skipped={val_stats.boxes_skipped}, empty={val_stats.empty_labels}"
    )

    test_count = prepare_test_split(
        source_root,
        target_root,
        source_root / "wider_face_split" / "wider_face_test_filelist.txt",
    )
    print(f"test  -> images={test_count}, labels=tidak tersedia secara resmi")

    write_notes(target_root, train_stats, val_stats, test_count)
    validate_counts(train_stats, val_stats, test_count)
    print("Konversi WIDER ke format YOLO selesai.")


if __name__ == "__main__":
    main()
