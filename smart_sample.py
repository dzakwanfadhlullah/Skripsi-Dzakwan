import os
import shutil
import random
import math

def smart_sample(source_dir, target_dir, max_size_gb=2.5):
    """
    Sampling data membatasi size (GB).
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source {source_dir} tidak ditemukan.")
        return

    os.makedirs(target_dir, exist_ok=True)
    
    # Mencari semua file gambar (asumsi jpg/png)
    all_files = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))
    
    if not all_files:
        print("Tidak ada file gambar ditemukan.")
        return

    random.shuffle(all_files) # Shuffle agar data variatif
    
    current_size_bytes = 0
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024
    count = 0
    
    print(f"Sampling... Target: {max_size_gb}GB")
    
    for img_path in all_files:
        file_size = os.path.getsize(img_path)
        
        if current_size_bytes + file_size > max_size_bytes:
            break
            
        # Tentukan lokasi label (asumsi folder 'labels' sejajar atau ada mapping tertentu)
        # Untuk YOLO, biasanya image di 'images/' dan label di 'labels/'
        # Kita coba copy gambarnya dulu
        rel_path = os.path.relpath(img_path, source_dir)
        dest_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        shutil.copy2(img_path, dest_path)
        
        # Coba cari file label .txt yang sesuai
        # Strategi 1: Di folder yang sama
        label_path = img_path.rsplit('.', 1)[0] + '.txt'
        
        # Strategi 2: Cari di seluruh source_dir jika tidak ada di tempat (untuk dataset yang pisah folder labels)
        if not os.path.exists(label_path):
            # search logic here
            pass
            
        if os.path.exists(label_path):
            dest_label = dest_path.rsplit('.', 1)[0] + '.txt'
            shutil.copy2(label_path, dest_label)
            
        current_size_bytes += file_size
        count += 1
        
        if count % 500 == 0:
            print(f"Sudah memproses {count} gambar... ({current_size_bytes / (1024**3):.2f} GB)")

    print(f"Selesai! {count} files ({current_size_bytes / (1024**3):.2f} GB) disalin ke {target_dir}")

if __name__ == "__main__":
    # Contoh penggunaan (Akan disesuaikan di Colab)
    # smart_sample("/content/raw_data", "/content/drive/MyDrive/skripshit/datasets/sampled_data")
    pass
