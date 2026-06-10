"""
Visual Comparison: Side-by-Side Detection.
"""
import os
import cv2
import random
import numpy as np
from ultralytics import YOLO


# ==================== KONFIGURASI ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "skripsi_yolo")
VAL_LOWLIGHT = os.path.join(DATASET_DIR, "images", "val_lowlight")
VAL_NORMAL = os.path.join(DATASET_DIR, "images", "val")
LABEL_DIR = os.path.join(DATASET_DIR, "labels", "val")

BASELINE_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "yolov8_baseline_final", "weights", "best.pt")
AUGMENTED_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "runs", "detect", "yolov8_augmented", "weights", "best.pt")

OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results", "visual_comparison")
CONF_THRESHOLD = 0.25
NUM_SAMPLES = 12
SEED = 42
# =====================================================


def draw_detections(img, results, color, label_prefix=""):
    """Gambar bounding box dan confidence pada image."""
    boxes = results[0].boxes
    count = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label_prefix}{conf:.2f}"
        # Background untuk text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        count += 1
    return img, count


def draw_ground_truth(img, label_path, img_shape, color=(0, 255, 0)):
    """Gambar ground truth bounding box dari label YOLO."""
    h, w = img_shape[:2]
    count = 0
    if not os.path.exists(label_path):
        return img, 0
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            count += 1
    return img, count


def create_comparison_image(img_name, baseline_model, augmented_model):
    """Buat satu gambar perbandingan: Original | Baseline Detect | Augmented Detect."""
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(LABEL_DIR, base_name + ".txt")

    # Baca gambar low-light
    lowlight_path = os.path.join(VAL_LOWLIGHT, img_name)
    img_low = cv2.imread(lowlight_path)
    if img_low is None:
        return None, {}

    # Baca gambar normal (untuk referensi)
    normal_path = os.path.join(VAL_NORMAL, img_name)
    img_normal = cv2.imread(normal_path)

    # Hitung ground truth
    _, gt_count = draw_ground_truth(img_low.copy(), label_path, img_low.shape)

    # Deteksi dengan baseline
    img_baseline = img_low.copy()
    res_baseline = baseline_model(img_low.copy(), conf=CONF_THRESHOLD, verbose=False)
    img_baseline, baseline_count = draw_detections(img_baseline, res_baseline, (0, 0, 255), "B:")

    # Deteksi dengan augmented
    img_augmented = img_low.copy()
    res_augmented = augmented_model(img_low.copy(), conf=CONF_THRESHOLD, verbose=False)
    img_augmented, augmented_count = draw_detections(img_augmented, res_augmented, (255, 165, 0), "A:")

    # Resize semua ke tinggi yang sama
    target_h = 320
    def resize(img):
        aspect = img.shape[1] / img.shape[0]
        return cv2.resize(img, (int(target_h * aspect), target_h))

    panels = []

    # Panel 1: Gambar normal (referensi)
    if img_normal is not None:
        p1 = resize(img_normal)
        cv2.putText(p1, "NORMAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        panels.append(p1)

    # Panel 2: Gambar low-light + ground truth
    img_gt = img_low.copy()
    img_gt, _ = draw_ground_truth(img_gt, label_path, img_gt.shape, (0, 255, 0))
    p2 = resize(img_gt)
    cv2.putText(p2, f"LOW-LIGHT (GT: {gt_count})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    panels.append(p2)

    # Panel 3: Deteksi baseline (merah)
    p3 = resize(img_baseline)
    cv2.putText(p3, f"BASELINE ({baseline_count}/{gt_count})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    panels.append(p3)

    # Panel 4: Deteksi augmented (oranye)
    p4 = resize(img_augmented)
    cv2.putText(p4, f"AUGMENTED ({augmented_count}/{gt_count})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    panels.append(p4)

    # Samakan lebar semua panel
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)

    combined = np.hstack(padded)

    stats = {
        "gt": gt_count,
        "baseline": baseline_count,
        "augmented": augmented_count,
    }
    return combined, stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

    # Load models
    print("Loading models...")
    baseline_model = YOLO(BASELINE_WEIGHTS)
    augmented_model = YOLO(AUGMENTED_WEIGHTS)

    # Pilih sampel random
    all_images = [f for f in os.listdir(VAL_LOWLIGHT)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    samples = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))

    print(f"\nGenerating {len(samples)} comparison images...")
    all_stats = []

    for i, img_name in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {img_name}")
        combined, stats = create_comparison_image(img_name, baseline_model, augmented_model)
        if combined is not None:
            out_path = os.path.join(OUTPUT_DIR, f"compare_{i+1:02d}_{img_name}")
            cv2.imwrite(out_path, combined)
            stats["filename"] = img_name
            all_stats.append(stats)

    # Summary
    print("\n[Visual Comparison Summary]")
    total_gt = sum(s["gt"] for s in all_stats)
    total_baseline = sum(s["baseline"] for s in all_stats)
    total_augmented = sum(s["augmented"] for s in all_stats)
    print(f"  Total ground truth faces : {total_gt}")
    print(f"  Baseline detections      : {total_baseline} ({total_baseline/total_gt*100:.1f}%)")
    print(f"  Augmented detections     : {total_augmented} ({total_augmented/total_gt*100:.1f}%)")
    print(f"\n  Output saved to: {OUTPUT_DIR}")

    # Buat grid overview (2 kolom × N baris)
    print("\nGenerating overview grid...")
    all_comparisons = []
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("compare_") and f.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(OUTPUT_DIR, f))
            if img is not None:
                # Resize untuk grid
                target_w = 1600
                aspect = img.shape[0] / img.shape[1]
                img = cv2.resize(img, (target_w, int(target_w * aspect)))
                all_comparisons.append(img)

    if all_comparisons:
        # Stack vertikal dengan separator
        separator = np.ones((4, all_comparisons[0].shape[1], 3), dtype=np.uint8) * 200
        rows = []
        for comp in all_comparisons:
            rows.append(comp)
            rows.append(separator)
        grid = np.vstack(rows[:-1])  # remove last separator
        cv2.imwrite(os.path.join(OUTPUT_DIR, "OVERVIEW_GRID.jpg"), grid)
        print(f"  Overview grid saved: OVERVIEW_GRID.jpg")

    print("\nVisual comparison completed.")


if __name__ == "__main__":
    main()
