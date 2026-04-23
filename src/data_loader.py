import os
import cv2
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "OCT2017", "train", "NORMAL")

print("Current working dir:", os.getcwd())
print("Looking for:", os.path.abspath(DATA_PATH))
print("Exists?", os.path.exists(DATA_PATH))

def load_images(path, n=5):
    images = []
    files = os.listdir(path)[:n]

    for file in files:
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            images.append(img)

    return images

def show_images(images):
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    imgs = load_images(DATA_PATH)
    print("Number of images loaded:", len(imgs))
    print("Shape:", imgs[0].shape)

    show_images(imgs)