import os
import cv2
import numpy as np

INPUT_PATH = "data/raw/OCT2017/train/NORMAL"
OUTPUT_A = "data/processed/domain_A"
OUTPUT_B = "data/processed/domain_B"

os.makedirs(OUTPUT_A, exist_ok=True)
os.makedirs(OUTPUT_B, exist_ok=True)


# --------- Degradations ---------

def add_speckle_noise(image):
    noise = np.random.randn(*image.shape)
    noisy = image + image * noise * 0.2
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_gaussian_noise(image):
    noise = np.random.normal(0, 10, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def reduce_contrast(image):
    alpha = 0.6
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def degrade(image):
    img = add_speckle_noise(image)
    img = add_gaussian_noise(img)
    img = add_blur(img)
    img = reduce_contrast(img)
    return img


# --------- Main ---------

def process(n=100):
    files = os.listdir(INPUT_PATH)[:n]

    for i, file in enumerate(files):
        path = os.path.join(INPUT_PATH, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        degraded = degrade(img)

        cv2.imwrite(os.path.join(OUTPUT_A, f"A_{i}.png"), degraded)
        cv2.imwrite(os.path.join(OUTPUT_B, f"B_{i}.png"), img)

    print(f"{len(files)} images processed.")


if __name__ == "__main__":
    process(100)