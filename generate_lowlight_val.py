"""
Generate a low-light evaluation split from a clean source split.

Default behaviour:
- source split : test
- target split : test_lowlight
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from augment_data import augment_file_to_low_light
from project_config import IMAGE_EXTENSIONS, WIDER_CLEAN_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic low-light evaluation split from a clean split."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(WIDER_CLEAN_DIR),
        help="Root dataset clean yang akan diberi split low-light.",
    )
    parser.add_argument(
        "--source-split",
        default="test",
        help="Split sumber yang akan digelapkan.",
    )
    parser.add_argument(
        "--target-split",
        default="test_lowlight",
        help="Nama split hasil low-light.",
    )
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--brightness", type=int, default=-30)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Hapus target split jika sudah ada.",
    )
    return parser.parse_args()


def resolve_manifest_entry(dataset_root: Path, raw_line: str) -> Path:
    path = Path(raw_line.strip())
    if not path.is_absolute():
        path = (dataset_root / path).resolve()
    return path


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise FileNotFoundError(f"Tidak bisa menginfer label dari image path: {image_path}")


def load_source_images(dataset_root: Path, split_name: str) -> list[Path]:
    image_dir = dataset_root / "images" / split_name
    if image_dir.exists():
        return sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    manifest_path = dataset_root / f"{split_name}.txt"
    if manifest_path.exists():
        lines = [
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [resolve_manifest_entry(dataset_root, line) for line in lines]

    raise FileNotFoundError(f"Source split {split_name} tidak ditemukan di {dataset_root}")


def write_manifest(dataset_root: Path, split_name: str, image_paths: list[Path]) -> None:
    manifest_path = dataset_root / f"{split_name}.txt"
    manifest_path.write_text(
        "\n".join(str(path.resolve()) for path in image_paths) + "\n",
        encoding="utf-8",
    )


def generate_lowlight_split(
    dataset_root: Path,
    source_split: str,
    target_split: str,
    gamma: float,
    brightness: int,
    clear_target: bool = False,
) -> None:
    dst_img_dir = dataset_root / "images" / target_split
    dst_lbl_dir = dataset_root / "labels" / target_split
    dst_manifest = dataset_root / f"{target_split}.txt"

    source_images = load_source_images(dataset_root, source_split)
    if any(path.name.startswith("low_") for path in source_images):
        raise ValueError(
            f"Source split {source_split} sudah mengandung file low-light. "
            "Gunakan split clean sebagai sumber."
        )

    if clear_target:
        shutil.rmtree(dst_img_dir, ignore_errors=True)
        shutil.rmtree(dst_lbl_dir, ignore_errors=True)
        dst_manifest.unlink(missing_ok=True)

    shutil.rmtree(dst_img_dir, ignore_errors=True)
    shutil.rmtree(dst_lbl_dir, ignore_errors=True)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {target_split} from {source_split} "
        f"(Gamma={gamma}, Brightness={brightness})"
    )
    print(f"Total files: {len(source_images)}")

    generated_paths: list[Path] = []
    for image_path in tqdm(source_images, desc=f"Generating {target_split}"):
        label_path = infer_label_path(image_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Label tidak ditemukan untuk {image_path.name}")

        output_path = dst_img_dir / image_path.name
        ok = augment_file_to_low_light(
            image_path,
            output_path,
            gamma=gamma,
            brightness=brightness,
        )
        if not ok:
            raise RuntimeError(f"Gagal augment gambar {image_path}")

        shutil.copy2(label_path, dst_lbl_dir / label_path.name)
        generated_paths.append(output_path.resolve())

    write_manifest(dataset_root, target_split, generated_paths)
    print(f"Selesai membuat split {target_split} di {dataset_root}")


def main() -> None:
    args = parse_args()
    generate_lowlight_split(
        dataset_root=Path(args.dataset_root).resolve(),
        source_split=args.source_split,
        target_split=args.target_split,
        gamma=args.gamma,
        brightness=args.brightness,
        clear_target=args.clear,
    )


if __name__ == "__main__":
    main()
