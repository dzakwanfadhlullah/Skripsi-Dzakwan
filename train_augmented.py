from ultralytics import YOLO

from audit_dataset import audit_augmented_dataset
from project_config import (
    AUGMENTED_DATA_YAML,
    AUGMENTED_RUN_NAME,
    COMMON_TRAIN_ARGS,
    DEFAULT_MODEL_VARIANT,
    WIDER_AUG_DIR,
    WIDER_CLEAN_DIR,
)


def train_augmented():
    audit_augmented_dataset(WIDER_CLEAN_DIR, WIDER_AUG_DIR)

    model = YOLO(DEFAULT_MODEL_VARIANT)
    print("Starting augmented training...")

    train_args = dict(COMMON_TRAIN_ARGS)
    train_args.update(
        {
            "data": str(AUGMENTED_DATA_YAML),
            "name": AUGMENTED_RUN_NAME,
        }
    )
    results = model.train(**train_args)

    print(f"Done. Best weights: runs/detect/{AUGMENTED_RUN_NAME}/weights/best.pt")
    return results


if __name__ == "__main__":
    train_augmented()
