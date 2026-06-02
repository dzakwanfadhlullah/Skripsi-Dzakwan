import cv2
import numpy as np
from pathlib import Path


def adjust_gamma(image, gamma=0.4):
    """Gamma correction untuk simulasi low-light."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)],
        dtype="uint8",
    )
    return cv2.LUT(image, table)


def adjust_brightness(image, value=-30):
    """Adjust brightness pada channel V di HSV."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def augment_image_to_low_light(image, gamma=0.4, brightness=-30):
    """Pipeline augmentasi low-light: gamma correction lalu brightness reduction."""
    img_low = adjust_gamma(image, gamma=gamma)
    img_low = adjust_brightness(img_low, value=brightness)
    return img_low


def augment_file_to_low_light(src_path, dst_path, gamma=0.4, brightness=-30):
    """Augment satu file gambar ke low-light dan simpan ke file target."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    image = cv2.imread(str(src_path))
    if image is None:
        return False

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img_low = augment_image_to_low_light(image, gamma=gamma, brightness=brightness)
    cv2.imwrite(str(dst_path), img_low)
    return True


if __name__ == "__main__":
    pass
