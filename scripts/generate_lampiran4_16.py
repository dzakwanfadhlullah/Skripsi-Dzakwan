# Pipeline 16: generator materi Lampiran 4.
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random

ROOT_DIR = Path(__file__).resolve().parents[1]

TEST_MANIFEST = str(ROOT_DIR / "datasets" / "experiment" / "wider_face_clean" / "test.txt")
LOWLIGHT_DIR = str(ROOT_DIR / "datasets" / "experiment" / "wider_face_clean" / "images" / "test_lowlight")
OUT_PNG = str(ROOT_DIR / "thesis" / "appendices" / "lampiran_4_sampel_hasil_augmentasi_lowlight.png")
OUT_TXT = str(ROOT_DIR / "thesis" / "appendices" / "lampiran_4_sampel_hasil_augmentasi_lowlight.txt")

def get_label_path(img_path):
    return img_path.replace("\\images\\", "\\labels\\").replace(".jpg", ".txt").replace(".png", ".txt")

def draw_boxes(img_path, lbl_path, draw=True):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if draw and os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, cx, cy, bw, bh = map(float, parts[:5])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return img_rgb

# Read manifest and pick 3 samples
with open(TEST_MANIFEST, "r", encoding="utf-8") as f:
    all_paths = [line.strip() for line in f if line.strip()]

rng = random.Random(42)
indices = sorted(rng.sample(range(len(all_paths)), min(100, len(all_paths))))
candidates = [all_paths[i] for i in indices]

selected_normal = []
selected_lowlight = []
selected_labels = []

for normal_path in candidates:
    fname = os.path.basename(normal_path)
    lowlight_path = os.path.join(LOWLIGHT_DIR, fname)
    label_path = get_label_path(normal_path)

    if os.path.exists(normal_path) and os.path.exists(lowlight_path):
        img = cv2.imread(normal_path)
        if img is not None:
            h, w = img.shape[:2]
            # Prefer somewhat large landscape images for nicer look
            if w > h and w >= 300:
                selected_normal.append(normal_path)
                selected_lowlight.append(lowlight_path)
                selected_labels.append(label_path)

    if len(selected_normal) == 3:
        break

n_samples = len(selected_normal)
fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4.2 * n_samples), facecolor='white')

for i in range(n_samples):
    norm_path = selected_normal[i]
    low_path = selected_lowlight[i]
    lbl_path = selected_labels[i]

    # Baris 1 & 2 (indeks 0 & 1) digambar bounding boxnya. Baris 3 (indeks 2) tanpa bounding box.
    img_norm = draw_boxes(norm_path, lbl_path, draw=(i < 2))
    img_low = draw_boxes(low_path, lbl_path, draw=(i < 2))

    axes[i, 0].imshow(img_norm)
    axes[i, 0].axis('off')

    axes[i, 1].imshow(img_low)
    axes[i, 1].axis('off')

    if i == 0:
        axes[i, 0].set_title("Citra Normal", fontsize=14, fontweight='bold', pad=15)
        axes[i, 1].set_title("Citra Hasil Augmentasi Low-Light", fontsize=14, fontweight='bold', pad=15)

# Atur tata letak dengan menyisakan ruang di bagian bawah untuk catatan kaki
plt.tight_layout(rect=[0, 0.06, 1, 1])

# Tambahkan catatan kaki di bawah gambar
fig.text(0.5, 0.02, "Bounding box ditampilkan pada sebagian sampel untuk menunjukkan anotasi tetap valid setelah augmentasi.",
         ha="center", va="bottom", fontsize=10.5, fontstyle="italic", color="#333333")

plt.savefig(OUT_PNG, dpi=250, bbox_inches='tight')
plt.close()

# Generate TXT
normal_list = "\n".join(selected_normal)
lowlight_list = "\n".join(selected_lowlight)
label_list = "\n".join(selected_labels)

txt_content = f"""Lampiran 4 Sampel Hasil Data Augmentation Low-Light

Lokasi gambar kolase:
{OUT_PNG}

Lokasi citra normal:
{normal_list}

Lokasi citra low-light:
{lowlight_list}

Lokasi file anotasi / label:
{label_list}

Lokasi file .txt / manifest terkait:
{TEST_MANIFEST}
"""

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(txt_content)

print("Done")
