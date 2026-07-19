# Pipeline 14: demonstrasi kualitatif melalui kamera.
"""
Demo Kualitatif Kamera: Skenario Baseline vs Skenario Augmented
================================================================
Demo inferensi real-time untuk sidang skripsi.
Membandingkan YOLOv8n yang dilatih pada dua skenario:
  - Baseline  : YOLOv8n dilatih pada data WIDER FACE normal
  - Augmented : YOLOv8n dilatih pada data normal + sintetis low-light

PENTING: Ini adalah demo kualitatif, BUKAN evaluasi akurasi.
Bukti kuantitatif berasal dari evaluasi test set (evaluate_robustness_12.py).
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from project_config import AUGMENTED_WEIGHTS, BASELINE_WEIGHTS, ROOT_DIR

# ==================== KONFIGURASI ====================
BASE_DIR = ROOT_DIR

# Sumber video: "webcam" atau "video"
SOURCE_MODE = "webcam"
VIDEO_PATH = "demo_assets/lowlight_demo.mp4"

CAMERA_ID = 0
CONF_THRESHOLD = 0.25
WINDOW_NAME = "Skripsi Demo: Baseline vs Augmented (YOLOv8n)"

# Warna panel (BGR)
COLOR_BASELINE = (0, 0, 255)       # Merah
COLOR_AUGMENTED = (200, 180, 0)    # Teal/Cyan
COLOR_FROZEN = (0, 200, 255)       # Kuning/Oranye

# Level simulasi low-light
# Level 2 = default skripsi (gamma=0.4, brightness=-30)
LOW_LIGHT_LEVELS = {
    0: {"name": "Normal",           "gamma": 1.0,  "brightness": 0},
    1: {"name": "Low-light Ringan", "gamma": 0.7,  "brightness": -15},
    2: {"name": "Low-light Sedang", "gamma": 0.4,  "brightness": -30},
    3: {"name": "Low-light Berat",  "gamma": 0.25, "brightness": -45},
}

# Hasil kuantitatif dari evaluasi test set (untuk ditampilkan sebagai referensi)
THESIS_RESULTS = {
    "map50_lowlight_baseline": 0.4505,
    "map50_lowlight_augmented": 0.5541,
    "robustness_drop_baseline": 32.82,
    "robustness_drop_augmented": 18.05,
}

SCREENSHOT_DIR = BASE_DIR / "evaluation_results" / "camera_screenshots"
# =====================================================


# ==================== FUNGSI UTILITAS ================

def adjust_gamma(img, gamma=0.4):
    """Gamma correction untuk simulasi low-light."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)


def adjust_brightness(img, value=-30):
    """Adjust brightness pada channel V di HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def apply_low_light(img, gamma, brightness):
    """Terapkan pipeline simulasi low-light: gamma correction lalu brightness adjustment."""
    if gamma == 1.0 and brightness == 0:
        return img
    result = adjust_gamma(img, gamma=gamma)
    result = adjust_brightness(result, value=brightness)
    return result


def run_inference(model, frame, conf):
    """Jalankan inferensi dan kembalikan results beserta waktu (ms)."""
    t0 = time.perf_counter()
    results = model(frame.copy(), conf=conf, verbose=False)
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0
    return results, infer_ms


def extract_metrics(results):
    """Ekstrak metrik deteksi dari YOLO results."""
    boxes = results[0].boxes
    confs = [box.conf[0].item() for box in boxes]
    count = len(confs)
    avg_conf = sum(confs) / count if count > 0 else 0.0
    max_conf = max(confs) if count > 0 else 0.0
    return {
        "count": count,
        "avg_conf": avg_conf,
        "max_conf": max_conf,
        "confidences": confs,
    }


def draw_boxes(img, results, color):
    """Gambar bounding box pada image. Kembalikan image dan jumlah deteksi."""
    boxes = results[0].boxes
    count = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Label confidence
        text = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        count += 1
    return img, count


def draw_panel_header(img, title, subtitle, color):
    """Gambar header panel dengan judul dan subjudul skenario."""
    h, w = img.shape[:2]
    # Background header
    cv2.rectangle(img, (0, 0), (w, 52), (20, 20, 30), -1)
    # Judul utama
    cv2.putText(img, title, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    # Subjudul
    cv2.putText(img, subtitle, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def draw_panel_metrics(img, metrics, infer_ms, y_offset=58):
    """Gambar ringkasan metrik di sudut kiri panel."""
    lines = [
        f"Faces: {metrics['count']}",
        f"Avg Conf: {metrics['avg_conf']:.2f}",
        f"Max Conf: {metrics['max_conf']:.2f}",
        f"Infer: {infer_ms:.0f} ms",
    ]
    for i, line in enumerate(lines):
        y = y_offset + i * 18
        # Background semi-transparan
        cv2.rectangle(img, (4, y - 13), (160, y + 4), (0, 0, 0), -1)
        cv2.putText(img, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1,
                    cv2.LINE_AA)
    return img


def draw_status_bar(combined, ll_level, ll_info, conf, total_fps, is_frozen):
    """Gambar status bar di bagian atas combined frame."""
    h, w = combined.shape[:2]
    bar_h = 28
    cv2.rectangle(combined, (0, 0), (w, bar_h), (15, 15, 20), -1)

    # Info low-light
    if ll_level == 0:
        mode_str = "Normal"
    else:
        mode_str = f"{ll_info['name']} (gamma={ll_info['gamma']}, bright={ll_info['brightness']})"
    cv2.putText(combined, f"Mode: {mode_str}", (10, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Conf threshold
    cv2.putText(combined, f"Conf: {conf:.2f}", (w // 2 - 50, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Total FPS (seluruh pipeline)
    fps_text = f"Total FPS: {total_fps:.0f}"
    cv2.putText(combined, fps_text, (w - 180, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)

    # Indikator frozen frame
    if is_frozen:
        cv2.putText(combined, "FROZEN FRAME", (w // 2 - 80, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_FROZEN, 2, cv2.LINE_AA)

    return combined


def draw_thesis_results(combined):
    """Gambar ringkasan hasil kuantitatif skripsi di bagian bawah."""
    h, w = combined.shape[:2]
    box_h = 58
    y_start = h - box_h

    # Background
    cv2.rectangle(combined, (0, y_start), (w, h), (15, 15, 20), -1)
    cv2.line(combined, (0, y_start), (w, y_start), (60, 60, 70), 1)

    r = THESIS_RESULTS
    line1 = (f"mAP@50 Low-Light Sintetis:  "
             f"Baseline {r['map50_lowlight_baseline']:.4f}  |  "
             f"Augmented {r['map50_lowlight_augmented']:.4f}")
    line2 = (f"Robustness Drop mAP@50:  "
             f"Baseline {r['robustness_drop_baseline']:.2f}%  |  "
             f"Augmented {r['robustness_drop_augmented']:.2f}%")
    note = "Angka kuantitatif berasal dari test set, bukan dari kamera.  |  Demo kualitatif inferensi real-time, bukan evaluasi akurasi."

    cv2.putText(combined, line1, (10, y_start + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(combined, line2, (10, y_start + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(combined, note, (10, y_start + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 140, 140), 1, cv2.LINE_AA)
    return combined


def draw_help_overlay(combined):
    """Gambar overlay bantuan kontrol keyboard."""
    h, w = combined.shape[:2]
    overlay = combined.copy()
    # Semi-transparent background
    cv2.rectangle(overlay, (w // 4, h // 6), (3 * w // 4, 5 * h // 6),
                  (20, 20, 30), -1)
    combined = cv2.addWeighted(overlay, 0.85, combined, 0.15, 0)

    lines = [
        "=== KONTROL KEYBOARD ===",
        "",
        "Q         : Keluar",
        "S         : Simpan screenshot + metadata",
        "L         : Toggle low-light ON/OFF",
        "0/1/2/3   : Pilih level low-light",
        "SPACE     : Freeze / unfreeze frame",
        "+/=       : Naikkan confidence threshold",
        "-/_       : Turunkan confidence threshold",
        "H         : Tampilkan / sembunyikan bantuan ini",
        "",
        "Level 0: Normal (gamma=1.0, bright=0)",
        "Level 1: Ringan (gamma=0.7, bright=-15)",
        "Level 2: Sedang (gamma=0.4, bright=-30)  [DEFAULT SKRIPSI]",
        "Level 3: Berat  (gamma=0.25, bright=-45)",
        "",
        "Tekan H untuk menutup.",
    ]
    x0 = w // 4 + 30
    y0 = h // 6 + 35
    for i, line in enumerate(lines):
        color = (255, 255, 255) if i == 0 else (200, 200, 200)
        font_scale = 0.50 if i == 0 else 0.42
        thickness = 2 if i == 0 else 1
        cv2.putText(combined, line, (x0, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
                    cv2.LINE_AA)
    return combined


def save_screenshot(combined, metadata, screenshot_dir):
    """Simpan screenshot (JPG) beserta metadata (JSON)."""
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"screenshot_{ts}.jpg"
    json_name = f"screenshot_{ts}.json"
    img_path = screenshot_dir / img_name
    json_path = screenshot_dir / json_name

    cv2.imwrite(str(img_path), combined)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return img_path, json_path


def open_video_source():
    """Buka sumber video berdasarkan konfigurasi. Fallback ke webcam jika file video tidak ada."""
    if SOURCE_MODE == "video":
        video_full = BASE_DIR / VIDEO_PATH
        if video_full.exists():
            print(f"Membuka file video: {video_full}")
            cap = cv2.VideoCapture(str(video_full))
            if cap.isOpened():
                return cap
            print(f"Gagal membuka file video. Fallback ke webcam.")
        else:
            print(f"File video tidak ditemukan: {video_full}. Fallback ke webcam.")

    print(f"Membuka webcam (ID={CAMERA_ID})...")
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# ==================== MAIN ===========================

def main():
    """Entry point demo kualitatif sidang skripsi."""
    print("=" * 64)
    print("  Demo Kualitatif: Baseline vs Augmented (YOLOv8n)")
    print("  Skripsi: Analisis Robustness Deteksi Wajah")
    print("=" * 64)
    print()
    print(f"  Baseline weights : {BASELINE_WEIGHTS}")
    print(f"  Augmented weights: {AUGMENTED_WEIGHTS}")
    print()

    # --- Validasi model weights ---
    if not BASELINE_WEIGHTS.exists():
        print(f"ERROR: Weights baseline tidak ditemukan: {BASELINE_WEIGHTS}")
        print("Jalankan training baseline terlebih dahulu.")
        return
    if not AUGMENTED_WEIGHTS.exists():
        print(f"ERROR: Weights augmented tidak ditemukan: {AUGMENTED_WEIGHTS}")
        print("Jalankan training augmented terlebih dahulu.")
        return

    # --- Load models ---
    print("Memuat model Baseline...")
    model_baseline = YOLO(str(BASELINE_WEIGHTS))
    print("Memuat model Augmented...")
    model_augmented = YOLO(str(AUGMENTED_WEIGHTS))
    print("Kedua model berhasil dimuat.\n")

    # --- Buka sumber video ---
    cap = open_video_source()
    if not cap.isOpened():
        print("ERROR: Tidak dapat membuka sumber video/kamera.")
        return

    # --- State variabel ---
    conf = CONF_THRESHOLD
    ll_level = 0                    # 0 = normal, 2 = default skripsi
    low_light_on = False
    is_frozen = False
    frozen_frame = None
    show_help = False
    screenshot_count = 0

    print("Kontrol: [Q] Keluar  [S] Screenshot  [L] Low-light  [0-3] Level")
    print("         [SPACE] Freeze  [+/-] Threshold  [H] Bantuan")
    print()

    while True:
        t_start = time.perf_counter()

        # --- Baca frame ---
        if not is_frozen:
            ret, raw_frame = cap.read()
            if not ret:
                # Untuk video: loop kembali
                if SOURCE_MODE == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_frame = cap.read()
                    if not ret:
                        print("Video habis.")
                        break
                else:
                    print("Kamera terputus.")
                    break
            frozen_frame = raw_frame.copy()
        else:
            raw_frame = frozen_frame.copy()

        # --- Terapkan simulasi low-light ---
        ll_info = LOW_LIGHT_LEVELS[ll_level if low_light_on else 0]
        frame = apply_low_light(raw_frame, ll_info["gamma"], ll_info["brightness"])

        # --- Inferensi kedua model pada frame yang SAMA ---
        res_bl, infer_bl_ms = run_inference(model_baseline, frame, conf)
        res_aug, infer_aug_ms = run_inference(model_augmented, frame, conf)

        # --- Ekstrak metrik ---
        metrics_bl = extract_metrics(res_bl)
        metrics_aug = extract_metrics(res_aug)

        # --- Gambar panel Baseline ---
        panel_bl = frame.copy()
        panel_bl, _ = draw_boxes(panel_bl, res_bl, COLOR_BASELINE)
        panel_bl = draw_panel_header(
            panel_bl,
            "SKENARIO BASELINE",
            "YOLOv8n | train normal",
            COLOR_BASELINE,
        )
        panel_bl = draw_panel_metrics(panel_bl, metrics_bl, infer_bl_ms)

        # --- Gambar panel Augmented ---
        panel_aug = frame.copy()
        panel_aug, _ = draw_boxes(panel_aug, res_aug, COLOR_AUGMENTED)
        panel_aug = draw_panel_header(
            panel_aug,
            "SKENARIO AUGMENTED",
            "YOLOv8n | normal + low-light",
            COLOR_AUGMENTED,
        )
        panel_aug = draw_panel_metrics(panel_aug, metrics_aug, infer_aug_ms)

        # --- Gabungkan side-by-side ---
        # Garis pemisah tipis antar panel
        separator = np.full((panel_bl.shape[0], 3, 3), (60, 60, 70), dtype=np.uint8)
        combined = np.hstack([panel_bl, separator, panel_aug])

        # --- Total FPS (seluruh pipeline: capture + 2 model + draw) ---
        t_end = time.perf_counter()
        total_fps = 1.0 / (t_end - t_start + 1e-9)

        # --- Status bar atas ---
        status_bar = np.zeros((28, combined.shape[1], 3), dtype=np.uint8)
        status_bar[:] = (15, 15, 20)
        combined = np.vstack([status_bar, combined])
        combined = draw_status_bar(
            combined, ll_level if low_light_on else 0, ll_info,
            conf, total_fps, is_frozen,
        )

        # --- Hasil kuantitatif skripsi (footer) ---
        combined = draw_thesis_results(combined)

        # --- Help overlay ---
        if show_help:
            combined = draw_help_overlay(combined)

        # --- Tampilkan ---
        cv2.imshow(WINDOW_NAME, combined)

        # --- Keyboard input ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif key == ord(' '):
            is_frozen = not is_frozen
            status = "AKTIF (frame dibekukan)" if is_frozen else "NONAKTIF"
            print(f"Freeze frame: {status}")

        elif key == ord('s') or key == ord('S'):
            screenshot_count += 1
            active_ll = ll_level if low_light_on else 0
            active_info = LOW_LIGHT_LEVELS[active_ll]
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "source_mode": SOURCE_MODE,
                "low_light_on": low_light_on,
                "low_light_level": active_ll,
                "low_light_name": active_info["name"],
                "gamma": active_info["gamma"],
                "brightness": active_info["brightness"],
                "confidence_threshold": round(conf, 2),
                "is_frozen": is_frozen,
                "baseline": {
                    "face_count": metrics_bl["count"],
                    "avg_confidence": round(metrics_bl["avg_conf"], 4),
                    "max_confidence": round(metrics_bl["max_conf"], 4),
                    "inference_time_ms": round(infer_bl_ms, 1),
                },
                "augmented": {
                    "face_count": metrics_aug["count"],
                    "avg_confidence": round(metrics_aug["avg_conf"], 4),
                    "max_confidence": round(metrics_aug["max_conf"], 4),
                    "inference_time_ms": round(infer_aug_ms, 1),
                },
                "total_fps": round(total_fps, 1),
                "note": "Data kualitatif dari demo real-time, bukan evaluasi akurasi test set.",
            }
            img_path, json_path = save_screenshot(combined, metadata, SCREENSHOT_DIR)
            print(f"Screenshot #{screenshot_count} disimpan:")
            print(f"  Image: {img_path}")
            print(f"  Meta : {json_path}")

        elif key == ord('l') or key == ord('L'):
            low_light_on = not low_light_on
            if low_light_on and ll_level == 0:
                ll_level = 2  # Default skripsi saat pertama kali diaktifkan
            status = "ON" if low_light_on else "OFF"
            info = LOW_LIGHT_LEVELS[ll_level if low_light_on else 0]
            print(f"Low-light: {status} | Level {ll_level}: {info['name']} "
                  f"(gamma={info['gamma']}, brightness={info['brightness']})")

        elif key == ord('h') or key == ord('H'):
            show_help = not show_help

        elif key in (ord('0'), ord('1'), ord('2'), ord('3')):
            ll_level = key - ord('0')
            if ll_level > 0:
                low_light_on = True
            else:
                low_light_on = False
            info = LOW_LIGHT_LEVELS[ll_level]
            print(f"Low-light level {ll_level}: {info['name']} "
                  f"(gamma={info['gamma']}, brightness={info['brightness']})")

        elif key == ord('+') or key == ord('='):
            conf = min(0.90, round(conf + 0.05, 2))
            print(f"Confidence threshold: {conf:.2f}")

        elif key == ord('-') or key == ord('_'):
            conf = max(0.05, round(conf - 0.05, 2))
            print(f"Confidence threshold: {conf:.2f}")

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo kamera ditutup.")


if __name__ == "__main__":
    main()
