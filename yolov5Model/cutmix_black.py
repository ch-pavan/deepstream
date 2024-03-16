import os
import random
import cv2
import numpy as np

def load_image(image_path):
    # Load image
    return cv2.imread(image_path)

def black_out_area(image):
    h, w = image.shape[:2]
    area = h * w

    # Calculate 10% of the image area
    blackout_area = 0.1 * area
    blackout_width = random.randint(1, w)
    blackout_height = int(blackout_area / blackout_width)
    
    # Adjust height if it exceeds image height
    if blackout_height > h:
        blackout_height = h
        blackout_width = int(blackout_area / h)

    # Choose a random starting point for the blackout area
    x_start = random.randint(0, w - blackout_width)
    y_start = random.randint(0, h - blackout_height)

    # Apply blackout
    image[y_start:y_start+blackout_height, x_start:x_start+blackout_width] = 0

def main():
    image_dir = "/home/ch/Downloads/yolov5Model/data/mouse/images"

    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for i, image_path in enumerate(images):
        image = load_image(image_path)
        black_out_area(image)
        cv2.imwrite(f'/home/ch/Downloads/yolov5Model/data/mouse/images/aug_{i}.jpg', image)

if __name__ == "__main__":
    main()
