"""
Dataset audit utilities to stop train/eval when leakage or count issues exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from project_config import (
    DARKFACE_EXTERNAL_SPLITS,
    DARK_CLEAN_DIR,
    EXPECTED_SPLITS,
    IMAGE_EXTENSIONS,
    WIDER_AUG_DIR,
    WIDER_CLEAN_DIR,
)


def canonical_stem(stem: str) -> str:
    while stem.startswith("low_"):
        stem = stem[4:]
    return stem


def assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise AssertionError(f"{description} tidak ditemukan: {path}")


def list_image_paths(split_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def manifest_path(dataset_root: Path, split_name: str) -> Path:
    return dataset_root / f"{split_name}.txt"


def resolve_path_from_manifest(dataset_root: Path, raw_line: str) -> Path:
    candidate = Path(raw_line.strip())
    if not candidate.is_absolute():
        candidate = (dataset_root / candidate).resolve()
    return candidate


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
        if parts[index] == "image":
            parts[index] = "label"
            return Path(*parts).with_suffix(".txt")
    raise AssertionError(f"Tidak bisa menginfer label dari path image: {image_path}")


def load_manifest_images(dataset_root: Path, split_name: str) -> list[Path]:
    path = manifest_path(dataset_root, split_name)
    assert_exists(path, f"Manifest {split_name}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [resolve_path_from_manifest(dataset_root, line) for line in lines]


def resolve_split_images(dataset_root: Path, split_name: str) -> list[Path]:
    if manifest_path(dataset_root, split_name).exists():
        return load_manifest_images(dataset_root, split_name)

    image_dir = dataset_root / "images" / split_name
    if image_dir.exists():
        return list_image_paths(image_dir)

    raise AssertionError(f"Split {split_name} tidak ditemukan di {dataset_root}")


def resolve_split_labels(dataset_root: Path, split_name: str, image_paths: list[Path]) -> list[Path]:
    if manifest_path(dataset_root, split_name).exists():
        return [infer_label_path(path) for path in image_paths]

    label_dir = dataset_root / "labels" / split_name
    if label_dir.exists():
        return sorted(
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        )
    return [infer_label_path(path) for path in image_paths]


def audit_clean_dataset(
    dataset_root: Path,
    dataset_name: str,
    require_test_lowlight: bool = False,
) -> dict[str, set[str]]:
    expected_splits = EXPECTED_SPLITS[dataset_name]
    split_ids: dict[str, set[str]] = {}

    for split_name, expected_count in expected_splits.items():
        image_paths = resolve_split_images(dataset_root, split_name)
        label_paths = resolve_split_labels(dataset_root, split_name, image_paths)
        image_ids = {path.stem for path in image_paths}
        label_ids = {path.stem for path in label_paths}

        if len(image_paths) != expected_count:
            raise AssertionError(
                f"{dataset_name}:{split_name} image count {len(image_paths)} "
                f"!= expected {expected_count}"
            )
        if len(label_paths) != expected_count:
            raise AssertionError(
                f"{dataset_name}:{split_name} label count {len(label_paths)} "
                f"!= expected {expected_count}"
            )
        if image_ids != label_ids:
            missing = sorted(image_ids - label_ids)
            orphan = sorted(label_ids - image_ids)
            raise AssertionError(
                f"{dataset_name}:{split_name} image-label mismatch. "
                f"missing={len(missing)}, orphan={len(orphan)}"
            )
        if any(not path.exists() for path in image_paths):
            raise AssertionError(f"{dataset_name}:{split_name} mengandung image path yang hilang.")
        if any(not path.exists() for path in label_paths):
            raise AssertionError(f"{dataset_name}:{split_name} mengandung label path yang hilang.")
        if any(name.startswith("low_") for name in image_ids):
            raise AssertionError(
                f"{dataset_name}:{split_name} harus clean, tapi mengandung prefix low_."
            )

        split_ids[split_name] = image_ids

    if split_ids["train"] & split_ids["val"]:
        raise AssertionError(f"{dataset_name} train dan val overlap.")
    if split_ids["train"] & split_ids["test"]:
        raise AssertionError(f"{dataset_name} train dan test overlap.")
    if split_ids["val"] & split_ids["test"]:
        raise AssertionError(f"{dataset_name} val dan test overlap.")

    if require_test_lowlight:
        image_paths = resolve_split_images(dataset_root, "test_lowlight")
        label_paths = resolve_split_labels(dataset_root, "test_lowlight", image_paths)
        image_ids = {path.stem for path in image_paths}
        label_ids = {path.stem for path in label_paths}

        if len(image_paths) != expected_splits["test"]:
            raise AssertionError(
                f"{dataset_name}:test_lowlight image count {len(image_paths)} "
                f"!= expected {expected_splits['test']}"
            )
        if image_ids != split_ids["test"]:
            raise AssertionError(
                f"{dataset_name}:test_lowlight canonical IDs tidak sama dengan test."
            )
        if image_ids != label_ids:
            raise AssertionError(f"{dataset_name}:test_lowlight image-label mismatch.")
        if any(not path.exists() for path in image_paths):
            raise AssertionError(f"{dataset_name}:test_lowlight mengandung image path yang hilang.")
        if any(not path.exists() for path in label_paths):
            raise AssertionError(f"{dataset_name}:test_lowlight mengandung label path yang hilang.")

        split_ids["test_lowlight"] = image_ids

    return split_ids


def audit_augmented_dataset(clean_root: Path, aug_root: Path) -> None:
    clean_ids = audit_clean_dataset(clean_root, "wider_face", require_test_lowlight=True)

    image_paths = resolve_split_images(aug_root, "train")
    label_paths = resolve_split_labels(aug_root, "train", image_paths)
    image_ids = {path.stem for path in image_paths}
    label_ids = {path.stem for path in label_paths}
    if image_ids != label_ids:
        raise AssertionError("Augmented train image-label mismatch.")
    if any(not path.exists() for path in image_paths):
        raise AssertionError("Augmented train mengandung image path yang hilang.")
    if any(not path.exists() for path in label_paths):
        raise AssertionError("Augmented train mengandung label path yang hilang.")

    nested = [path.name for path in image_paths if path.name.startswith("low_low_")]
    if nested:
        raise AssertionError(
            f"Augmented train mengandung nested low-light variant: {nested[:5]}"
        )

    original_ids = {path.stem for path in image_paths if not path.name.startswith("low_")}
    lowlight_ids = {
        canonical_stem(path.stem) for path in image_paths if path.name.startswith("low_")
    }
    clean_train_ids = clean_ids["train"]

    if original_ids != clean_train_ids:
        raise AssertionError("Augmented train original IDs tidak sama dengan clean train.")
    if lowlight_ids != clean_train_ids:
        raise AssertionError(
            "Augmented train low-light IDs harus 1:1 dengan clean train."
        )
    if len(image_paths) != len(clean_train_ids) * 2:
        raise AssertionError(
            "Augmented train harus berisi tepat 2x jumlah clean train "
            "(original + satu low-light per citra)."
        )

    for split_name in ("val", "test", "test_lowlight"):
        aug_images = resolve_split_images(aug_root, split_name)
        aug_labels = resolve_split_labels(aug_root, split_name, aug_images)
        aug_ids = {path.stem for path in aug_images}
        aug_lbl_ids = {path.stem for path in aug_labels}
        if aug_ids != aug_lbl_ids:
            raise AssertionError(f"Augmented {split_name} image-label mismatch.")
        if aug_ids != clean_ids[split_name]:
            raise AssertionError(
                f"Augmented {split_name} harus identik dengan clean {split_name}."
            )
        if split_name != "test_lowlight" and any(name.startswith("low_") for name in aug_ids):
            raise AssertionError(f"Augmented {split_name} harus bebas prefix low_.")

    train_canonical = {canonical_stem(path.stem) for path in image_paths}
    for split_name in ("val", "test"):
        eval_ids = {path.stem for path in resolve_split_images(aug_root, split_name)}
        if train_canonical & eval_ids:
            raise AssertionError(f"Augmented train overlap dengan {split_name}.")


def audit_darkface_external_dataset(dataset_root: Path) -> None:
    expected_count = DARKFACE_EXTERNAL_SPLITS["test"]
    image_paths = resolve_split_images(dataset_root, "test")
    label_paths = resolve_split_labels(dataset_root, "test", image_paths)
    image_ids = {path.stem for path in image_paths}
    label_ids = {path.stem for path in label_paths}

    if len(image_paths) != expected_count:
        raise AssertionError(
            f"dark_face:test image count {len(image_paths)} != expected {expected_count}"
        )
    if len(label_paths) != expected_count:
        raise AssertionError(
            f"dark_face:test label count {len(label_paths)} != expected {expected_count}"
        )
    if image_ids != label_ids:
        raise AssertionError("dark_face:test image-label mismatch.")
    if any(not path.exists() for path in image_paths):
        raise AssertionError("dark_face:test mengandung image path yang hilang.")
    if any(not path.exists() for path in label_paths):
        raise AssertionError("dark_face:test mengandung label path yang hilang.")


def run_full_audit(require_darkface: bool) -> None:
    audit_clean_dataset(WIDER_CLEAN_DIR, "wider_face", require_test_lowlight=True)
    audit_augmented_dataset(WIDER_CLEAN_DIR, WIDER_AUG_DIR)
    if require_darkface:
        audit_darkface_external_dataset(DARK_CLEAN_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anti-leak dataset audit.")
    parser.add_argument(
        "--require-darkface",
        action="store_true",
        help="Fail jika dataset Dark Face clean belum siap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_full_audit(require_darkface=args.require_darkface)
    print("Audit dataset lolos. Tidak ada leak, count mismatch, atau nested low-light.")


if __name__ == "__main__":
    main()
