import os
import random
import cv2
import numpy as np

def load_image_and_bboxes(image_path, label_path):
    # Load image
    image = cv2.imread(image_path)

    # Check if label file exists
    if not os.path.exists(label_path):
        return image, []

    # Load bounding boxes
    bboxes = []
    with open(label_path, 'r') as file:
        for line in file:
            class_label, x_center, y_center, width, height = map(float, line.split())
            bboxes.append((int(class_label), x_center, y_center, width, height))

    return image, bboxes

def apply_cutmix_to_bbox(image, bbox, other_images, label_dir, cutmix_percent):
    h, w = image.shape[:2]
    class_label, x_center, y_center, width, height = bbox

    # Convert normalized values to actual pixel values
    x_center, y_center, width, height = int(x_center * w), int(y_center * h), int(width * w), int(height * h)
    x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
    x2, y2 = x1 + width, y1 + height

    patch_width, patch_height = int(cutmix_percent * width), int(cutmix_percent * height)
    patch_x1, patch_y1 = random.randint(x1, x2 - patch_width), random.randint(y1, y2 - patch_height)
    patch_x2, patch_y2 = patch_x1 + patch_width, patch_y1 + patch_height

    # Select a random patch from another image
    other_image_path = random.choice(other_images)
    other_label_path = os.path.join(label_dir, os.path.basename(other_image_path).replace('.jpg', '.txt'))
    other_image, _ = load_image_and_bboxes(other_image_path, other_label_path)
    other_patch = other_image[patch_y1:patch_y2, patch_x1:patch_x2]

    # Replace the patch in the original image
    image[patch_y1:patch_y2, patch_x1:patch_x2] = other_patch

def main():
    image_dir = "/home/ch/Downloads/yolov5Model/data/mouse/images"
    label_dir = "/home/ch/Downloads/yolov5Model/data/mouse/labels"

    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for i, image_path in enumerate(images):
        label_path = os.path.join(label_dir, os.path.basename(image_path).replace('.jpg', '.txt'))
        image, bboxes = load_image_and_bboxes(image_path, label_path)

        for bbox in bboxes:
            class_label = bbox[0]
            cutmix_percent = 0.20 if class_label == 7 else 0.05 if class_label == 6 else 0  
            if cutmix_percent > 0:
                apply_cutmix_to_bbox(image, bbox, images, label_dir, cutmix_percent)

        cv2.imwrite(f'/home/ch/Downloads/yolov5Model/data/mouse/images/mix_{i}.jpg', image)

if __name__ == "__main__":
    main()
