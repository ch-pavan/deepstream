import os
import random
import cv2
import numpy as np

def load_image_and_bboxes(image_path, label_path):
    image = cv2.imread(image_path)

    if not os.path.exists(label_path):
        return image, []

    bboxes = []
    with open(label_path, 'r') as file:
        for line in file:
            class_label, x_center, y_center, width, height = map(float, line.split())
            bboxes.append((int(class_label), x_center, y_center, width, height))

    return image, bboxes

def apply_cutmix_to_bbox(image, bbox, other_image, patch_size):
    h, w = image.shape[:2]
    class_label, x_center, y_center, width, height = bbox

    x_center, y_center, width, height = int(x_center * w), int(y_center * h), int(width * w), int(height * h)
    x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
    x2, y2 = x1 + width, y1 + height
    patch_width, patch_height = int(patch_size * width), int(patch_size * height)
    patch_x1, patch_y1 = random.randint(x1, x2 - patch_width), random.randint(y1, y2 - patch_height)
    patch_x2, patch_y2 = patch_x1 + patch_width, patch_y1 + patch_height

    other_patch = other_image[patch_y1:patch_y2, patch_x1:patch_x2]

    image[patch_y1:patch_y2, patch_x1:patch_x2] = other_patch

def save_annotations(annotations, file_path):
    with open(file_path, 'w') as file:
        for annotation in annotations:
            file.write(' '.join(map(str, annotation)) + '\n')

def main():
    patch_size = 0.2
    image_dir = "/home/ch/Downloads/yolov5Model/data/images/train"
    label_dir = "/home/ch/Downloads/yolov5Model/data/labels/train"

    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    labels = [os.path.join(label_dir, os.path.splitext(os.path.basename(f))[0] + '.txt') for f in images]

    for i, image_path in enumerate(images):
        label_path = labels[i]
        image, bboxes = load_image_and_bboxes(image_path, label_path)
        other_image_path = random.choice(images)
        other_image, _ = load_image_and_bboxes(other_image_path, label_path.replace(image_path, other_image_path))

        for bbox in bboxes:
            apply_cutmix_to_bbox(image, bbox, other_image, patch_size)

        cv2.imwrite(f'/home/ch/Downloads/yolov5Model/data/images/train/aug_{i}.jpg', image)
        
        save_annotations(bboxes, f'/home/ch/Downloads/yolov5Model/data/labels/train/aug_{i}.txt')

if __name__ == "__main__":
    main()
