import cv2
import matplotlib.pyplot as plt
import os
import random

def visualize_sample(image_dir, label_dir, num_samples=3):
    """
    Menampilkan sampel gambar dengan bounding box untuk verifikasi label.
    """
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    samples = random.sample(images, min(num_samples, len(images)))
    
    plt.figure(figsize=(15, 10))
    
    for i, img_name in enumerate(samples):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        _, x_c, y_c, wb, hb = map(float, data)
                        # Convert YOLO format to pixel coordinates
                        x1 = int((x_c - wb/2) * w)
                        y1 = int((y_c - hb/2) * h)
                        x2 = int((x_c + wb/2) * w)
                        y2 = int((y_c + hb/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # visualize_sample("datasets/skripsi_yolo/images/train", "datasets/skripsi_yolo/labels/train")
    pass
