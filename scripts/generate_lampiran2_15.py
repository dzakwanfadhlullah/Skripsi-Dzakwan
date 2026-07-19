# Pipeline 15: generator materi Lampiran 2.
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

WIDER_IMG = str(ROOT_DIR / "datasets" / "raw" / "wider_face_yolo" / "images" / "train" / "0_Parade_Parade_0_127.jpg")
WIDER_LBL = str(ROOT_DIR / "datasets" / "raw" / "wider_face_yolo" / "labels" / "train" / "0_Parade_Parade_0_127.txt")

DARK_IMG = str(ROOT_DIR / "datasets" / "experiment" / "dark_face_clean" / "images" / "test" / "1.png")
DARK_LBL = str(ROOT_DIR / "datasets" / "experiment" / "dark_face_clean" / "labels" / "test" / "1.txt")

OUT_PNG = str(ROOT_DIR / "thesis" / "appendices" / "lampiran_2_sampel_dataset_anotasi.png")
OUT_TXT = str(ROOT_DIR / "thesis" / "appendices" / "lampiran_2_sampel_anotasi_yolo.txt")

def draw_boxes(img_path, lbl_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls, cx, cy, bw, bh = map(float, parts[:5])
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_rgb

# Generate images
img_wider = cv2.cvtColor(cv2.imread(WIDER_IMG), cv2.COLOR_BGR2RGB)
img_dark = cv2.cvtColor(cv2.imread(DARK_IMG), cv2.COLOR_BGR2RGB)
img_boxed = draw_boxes(WIDER_IMG, WIDER_LBL)

fig, axes = plt.subplots(3, 1, figsize=(7, 12), facecolor='white')

axes[0].imshow(img_wider)
axes[0].set_title("Sampel Citra WIDER FACE (Kondisi Normal)", fontsize=13, fontweight='bold', pad=12)
axes[0].axis('off')

axes[1].imshow(img_dark)
axes[1].set_title("Sampel Citra Dark Face (Kondisi Low-Light Eksternal)", fontsize=13, fontweight='bold', pad=12)
axes[1].axis('off')

axes[2].imshow(img_boxed)
axes[2].set_title("Contoh Citra WIDER FACE dengan Bounding Box YOLO", fontsize=13, fontweight='bold', pad=12)
axes[2].axis('off')

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=250, bbox_inches='tight')
plt.close()

# Generate TXT
with open(WIDER_LBL, "r") as f:
    lines = [line.strip() for line in f.readlines()[:5]]

txt_content = f"""Lampiran 2 Sampel Dataset dan Anotasi YOLO

Sampel citra WIDER FACE:
{WIDER_IMG}

Sampel citra Dark Face:
{DARK_IMG}

Sampel citra dengan bounding box:
{OUT_PNG}

Sampel file anotasi YOLO:
{WIDER_LBL}

Sampel isi anotasi YOLO:
{chr(10).join(lines)}
"""

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(txt_content)

print("Done")
