# Pipeline 01: download dataset mentah.
"""
Download raw Kaggle sources into datasets/raw for the clean experiment pipeline.

Important:
- The selected WIDER source now uses the official Google Drive image archives
  used by TensorFlow Datasets: `WIDER_train.zip`, `WIDER_val.zip`,
  and `WIDER_test.zip`.
- The official annotation archive is unstable for scripted download, so this
  script copies `wider_face_split/` from a Kaggle mirror that preserves the
  original official files.
- The selected Dark Face source may also require extra organization before it
  matches the exact thesis protocol.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import gdown
import kagglehub

from project_config import (
    DARK_KAGGLE_DATASET,
    DARK_RAW_DIR,
    WIDER_OFFICIAL_FILE_IDS,
    WIDER_RAW_DIR,
    WIDER_SPLIT_MIRROR_DATASET,
)


DATASET_SOURCES = {
    "dark_face": (
        DARK_KAGGLE_DATASET,
        DARK_RAW_DIR,
        [
            "Periksa lagi struktur hasil unduhan Dark Face sebelum menjalankan `organize_yolo_03.py`.",
            "Jika label YOLO dan manifest split belum ada, siapkan dulu sesuai protokol skripsi final.",
        ],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw Kaggle datasets into datasets/raw/."
    )
    parser.add_argument(
        "--dataset",
        choices=("wider_face", "dark_face", "all"),
        default="all",
        help="Dataset source to download.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the existing raw target folder before copying new contents.",
    )
    return parser.parse_args()


def copy_download_contents(download_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for item in download_root.iterdir():
        destination = target_root / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def write_download_note(
    target_root: Path,
    dataset_name: str,
    dataset_id: str,
    notes: list[str],
) -> None:
    note_path = target_root / "DOWNLOAD_INFO.txt"
    lines = [
        f"dataset_name={dataset_name}",
        f"kaggle_id={dataset_id}",
        "",
        "notes:",
        *[f"- {line}" for line in notes],
        "",
        "next_step:",
        "- Jalankan audit struktur raw sebelum menyiapkan split clean.",
    ]
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_split_placeholders(target_root: Path) -> None:
    splits_dir = target_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    readme_path = splits_dir / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "Isi folder ini dengan manifest split resmi.\n"
            "Setiap file train.txt / val.txt / test.txt harus berisi satu stem image per baris.\n",
            encoding="utf-8",
        )


def find_wider_split_mirror_dir(mirror_root: Path) -> Path:
    candidates = [
        mirror_root / "wider_face_split" / "wider_face_split",
        mirror_root / "wider_face_split",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Folder wider_face_split tidak ditemukan di mirror: {mirror_root}"
    )


def count_images(folder: Path) -> int:
    return sum(1 for _ in folder.rglob("*.jpg"))


def validate_wider_counts(target_root: Path) -> None:
    expected = {
        "train": 12880,
        "val": 3226,
        "test": 16097,
    }
    actual = {
        "train": count_images(target_root / "WIDER_train" / "images"),
        "val": count_images(target_root / "WIDER_val" / "images"),
        "test": count_images(target_root / "WIDER_test" / "images"),
    }
    if actual != expected:
        raise ValueError(
            f"Count resmi WIDER tidak cocok. expected={expected}, actual={actual}"
        )


def download_official_wider(clear_target: bool) -> None:
    target_root = WIDER_RAW_DIR
    archive_dir = target_root / "_downloads"
    notes = [
        "Image splits berasal dari source resmi WIDER FACE (train, val, test).",
        "Folder wider_face_split disalin dari mirror Kaggle `tngiaduc/widerface-dataset` karena URL anotasi resmi tidak stabil untuk otomasi.",
        "Struktur final raw mengikuti paket benchmark: WIDER_train/, WIDER_val/, WIDER_test/, wider_face_split/.",
    ]

    if clear_target and target_root.exists():
        shutil.rmtree(target_root)

    target_root.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading wider_face ===")
    print("Source    : Official WIDER FACE image archives")
    print(f"Target    : {target_root}")

    archive_names = {
        "train": "WIDER_train.zip",
        "val": "WIDER_val.zip",
        "test": "WIDER_test.zip",
    }
    for split_name, file_id in WIDER_OFFICIAL_FILE_IDS.items():
        archive_path = archive_dir / archive_names[split_name]
        print(f"Downloading {archive_names[split_name]}...")
        gdown.download(
            id=file_id,
            output=str(archive_path),
            quiet=False,
            resume=True,
        )
        print(f"Extracting {archive_names[split_name]}...")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_root)

    print("Copying wider_face_split mirror...")
    mirror_root = Path(kagglehub.dataset_download(WIDER_SPLIT_MIRROR_DATASET))
    mirror_split_dir = find_wider_split_mirror_dir(mirror_root)
    target_split_dir = target_root / "wider_face_split"
    shutil.rmtree(target_split_dir, ignore_errors=True)
    shutil.copytree(mirror_split_dir, target_split_dir)
    validate_wider_counts(target_root)
    shutil.rmtree(archive_dir, ignore_errors=True)
    write_download_note(
        target_root,
        "wider_face",
        "official_wider_face_images + tngiaduc/widerface-dataset:wider_face_split",
        notes,
    )

    print("Notes:")
    for note in notes:
        print(f"  - {note}")


def download_dataset(dataset_name: str, clear_target: bool) -> None:
    dataset_id, target_root, notes = DATASET_SOURCES[dataset_name]

    if clear_target and target_root.exists():
        shutil.rmtree(target_root)

    print(f"\n=== Downloading {dataset_name} ===")
    print(f"Kaggle ID : {dataset_id}")
    print(f"Target    : {target_root}")

    tmp_path = Path(kagglehub.dataset_download(dataset_id))
    print(f"Temp path : {tmp_path}")

    copy_download_contents(tmp_path, target_root)
    ensure_split_placeholders(target_root)
    write_download_note(target_root, dataset_name, dataset_id, notes)

    print("Notes:")
    for note in notes:
        print(f"  - {note}")


def main() -> None:
    args = parse_args()
    dataset_names = ["wider_face", "dark_face"] if args.dataset == "all" else [args.dataset]

    for dataset_name in dataset_names:
        if dataset_name == "wider_face":
            download_official_wider(clear_target=args.clear)
        else:
            download_dataset(dataset_name, clear_target=args.clear)

    print(
        "\nUnduhan selesai. Jangan langsung training. "
        "Pastikan split resmi dan label YOLO sudah valid sebelum menjalankan organize_yolo_03.py."
    )


if __name__ == "__main__":
    main()
