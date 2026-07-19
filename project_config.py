from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
EXPERIMENT_DATASETS_DIR = DATASETS_DIR / "experiment"

WIDER_RAW_DIR = RAW_DATASETS_DIR / "wider_face"
WIDER_YOLO_RAW_DIR = RAW_DATASETS_DIR / "wider_face_yolo"
DARK_RAW_DIR = RAW_DATASETS_DIR / "dark_face"

WIDER_SPLIT_MIRROR_DATASET = "tngiaduc/widerface-dataset"
DARK_KAGGLE_DATASET = "soumikrakshit/dark-face-dataset"
WIDER_OFFICIAL_FILE_IDS = {
    "train": "15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M",
    "val": "1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q",
    "test": "1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T",
}

WIDER_CLEAN_DIR = EXPERIMENT_DATASETS_DIR / "wider_face_clean"
WIDER_AUG_DIR = EXPERIMENT_DATASETS_DIR / "wider_face_augmented"
DARK_CLEAN_DIR = EXPERIMENT_DATASETS_DIR / "dark_face_clean"

RUNS_DIR = ROOT_DIR / "runs" / "detect"
EVAL_RESULTS_DIR = ROOT_DIR / "evaluation_results"

CONFIGS_DIR = ROOT_DIR / "configs"
MODELS_DIR = ROOT_DIR / "models"

BASELINE_DATA_YAML = CONFIGS_DIR / "wider_face_baseline.yaml"
AUGMENTED_DATA_YAML = CONFIGS_DIR / "wider_face_augmented.yaml"
DARKFACE_DATA_YAML = CONFIGS_DIR / "darkface_eval.yaml"

DEFAULT_MODEL_VARIANT = str(MODELS_DIR / "yolov8n.pt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

OFFICIAL_SPLITS = {
    "wider_face": {
        "train": 12880,
        "val": 3226,
        "test": 16097,
    },
    "dark_face": {
        "train": 5500,
        "val": 500,
        "test": 4000,
    },
}

DARKFACE_EXTERNAL_SPLITS = {
    "test": 6000,
}

WIDER_PROTOCOL_SEED = 42
WIDER_PROTOCOL_DEV_RATIO = 0.1
WIDER_PROTOCOL_SPLITS = {
    "train": 11592,
    "val": 1288,
    "test": 3226,
}

EXPECTED_SPLITS = {
    "wider_face": WIDER_PROTOCOL_SPLITS,
    "dark_face": OFFICIAL_SPLITS["dark_face"],
}

BASELINE_RUN_NAME = "yolov8n_baseline_clean3"
AUGMENTED_RUN_NAME = "yolov8n_augmented_clean"

# Selected runs for final evaluation/reporting.
# Keep these explicit so eval never silently falls back to stale artifacts.
SELECTED_BASELINE_RUN_NAME = BASELINE_RUN_NAME
SELECTED_AUGMENTED_RUN_NAME = AUGMENTED_RUN_NAME

BASELINE_WEIGHTS = RUNS_DIR / SELECTED_BASELINE_RUN_NAME / "weights" / "best.pt"
AUGMENTED_WEIGHTS = RUNS_DIR / SELECTED_AUGMENTED_RUN_NAME / "weights" / "best.pt"

COMMON_TRAIN_ARGS = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "save": True,
    "project": str(RUNS_DIR),
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "close_mosaic": 10,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "erasing": 0.4,
    "copy_paste": 0.0,
    "plots": True,
    "val": True,
    "patience": 100,
    "seed": 0,
    "deterministic": True,
}
