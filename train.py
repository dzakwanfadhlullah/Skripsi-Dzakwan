import os
from ultralytics import YOLO

def train_model(data_yaml_path, epochs=50, img_size=640, model_variant='yolov8n.pt'):
    print(f"Starting training with {model_variant}")
    
    # Load model
    model = YOLO(model_variant)
    
    # Jalankan training
    # Catatan: results akan disimpan otomatis di folder 'runs/detect/train'
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        device=0, # Menggunakan GPU pertama
        project='skripshit_research', # Nama project di wandb/folder
        name='baseline_training' # Nama subfolder hasil
    )
    
    print("Training finished.")
    return results

if __name__ == "__main__":
    # Path ini harus disesuaikan dengan lokasi di Google Drive/Colab nanti
    # Contoh: '/content/drive/MyDrive/skripshit/datasets/data.yaml'
    DATA_PATH = 'datasets/data.yaml' 
    
    if os.path.exists(DATA_PATH):
        train_model(DATA_PATH)
    else:
        print(f"Error: {DATA_PATH} not found.")
