"""
Live Camera Demo: Baseline vs Augmented
"""
import os
import cv2
import numpy as np
import time
from ultralytics import YOLO

from project_config import AUGMENTED_WEIGHTS, BASELINE_WEIGHTS


# ==================== KONFIGURASI ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONF_THRESHOLD = 0.25
CAMERA_ID = 0           # 0 = webcam utama
WINDOW_WIDTH = 1280      # Lebar total window
# =====================================================


def adjust_gamma(img, gamma=0.4):
    """Simulasi low-light dengan gamma correction."""
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def adjust_brightness(img, value=-30):
    """Kurangi brightness."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def draw_boxes(img, results, color, label=""):
    """Gambar bounding box pada image."""
    boxes = results[0].boxes
    count = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        count += 1
    return img, count


def draw_info_bar(img, text, color, count, fps=None):
    """Gambar info bar di atas image."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 36), (20, 20, 30), -1)
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img, f"Faces: {count}", (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if fps is not None:
        cv2.putText(img, f"FPS: {fps:.0f}", (w - 230, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    return img


def main():
    print("Starting Live Camera Demo...")
    print("Controls: [Q] Quit | [S] Screenshot | [L] Toggle Low-Light | [+/-] Threshold\n")
    print(f"Baseline weights : {BASELINE_WEIGHTS}")
    print(f"Augmented weights: {AUGMENTED_WEIGHTS}\n")

    if not BASELINE_WEIGHTS.exists():
        print(f"Error: baseline weights not found: {BASELINE_WEIGHTS}")
        return
    if not AUGMENTED_WEIGHTS.exists():
        print(f"Error: augmented weights not found: {AUGMENTED_WEIGHTS}")
        return

    # Load models
    print("Loading Baseline model...")
    baseline = YOLO(str(BASELINE_WEIGHTS))
    print("Loading Augmented model...")
    augmented = YOLO(str(AUGMENTED_WEIGHTS))
    print("Models loaded.\n")

    # Open camera
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    low_light_mode = False
    conf = CONF_THRESHOLD
    screenshot_count = 0
    screenshot_dir = os.path.join(BASE_DIR, "evaluation_results", "camera_screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    print("Camera started.\n")

    while True:
        t_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Terapkan simulasi low-light jika diaktifkan
        if low_light_mode:
            frame = adjust_gamma(frame, gamma=0.4)
            frame = adjust_brightness(frame, value=-30)

        # Deteksi dengan kedua model
        frame_baseline = frame.copy()
        frame_augmented = frame.copy()

        res_bl = baseline(frame.copy(), conf=conf, verbose=False)
        res_aug = augmented(frame.copy(), conf=conf, verbose=False)

        frame_baseline, count_bl = draw_boxes(frame_baseline, res_bl, (0, 0, 255))
        frame_augmented, count_aug = draw_boxes(frame_augmented, res_aug, (255, 165, 0))

        fps = 1.0 / (time.time() - t_start + 1e-6)

        # Info bars
        mode_text = " [LOW-LIGHT SIM]" if low_light_mode else ""
        frame_baseline = draw_info_bar(
            frame_baseline, f"BASELINE{mode_text}", (0, 0, 255), count_bl, fps)
        frame_augmented = draw_info_bar(
            frame_augmented, f"AUGMENTED{mode_text}", (255, 165, 0), count_aug)

        # Gabungkan side-by-side
        combined = np.hstack([frame_baseline, frame_augmented])

        # Tambah bottom bar
        h, w = combined.shape[:2]
        cv2.rectangle(combined, (0, h - 30), (w, h), (20, 20, 30), -1)
        controls = f"[Q]uit  [S]creenshot  [L]ow-light: {'ON' if low_light_mode else 'OFF'}  [+/-] Conf: {conf:.2f}"
        cv2.putText(combined, controls, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("YOLOv8 Face Detection: Baseline vs Augmented", combined)

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            screenshot_count += 1
            fname = f"screenshot_{screenshot_count:03d}.jpg"
            path = os.path.join(screenshot_dir, fname)
            cv2.imwrite(path, combined)
            print(f"Screenshot saved: {path}")
        elif key == ord('l') or key == ord('L'):
            low_light_mode = not low_light_mode
            status = "ON" if low_light_mode else "OFF"
            print(f"Low-light simulation: {status}")
        elif key == ord('+') or key == ord('='):
            conf = min(0.9, conf + 0.05)
            print(f"Conf threshold: {conf:.2f}")
        elif key == ord('-') or key == ord('_'):
            conf = max(0.05, conf - 0.05)
            print(f"Conf threshold: {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera demo closed.")


if __name__ == "__main__":
    main()
