# Pipeline 10: latih model baseline.
from ultralytics import YOLO

from scripts.audit_dataset_09 import audit_clean_dataset
from project_config import (
    BASELINE_DATA_YAML,
    BASELINE_RUN_NAME,
    COMMON_TRAIN_ARGS,
    DEFAULT_MODEL_VARIANT,
    WIDER_CLEAN_DIR,
)


def train_baseline():
    audit_clean_dataset(WIDER_CLEAN_DIR, "wider_face", require_test_lowlight=False)

    model = YOLO(DEFAULT_MODEL_VARIANT)
    print("Starting baseline training...")

    train_args = dict(COMMON_TRAIN_ARGS)
    train_args.update(
        {
            "data": str(BASELINE_DATA_YAML),
            "name": BASELINE_RUN_NAME,
        }
    )
    results = model.train(**train_args)

    print(f"Done. Best weights: runs/detect/{BASELINE_RUN_NAME}/weights/best.pt")
    return results


if __name__ == "__main__":
    train_baseline()
