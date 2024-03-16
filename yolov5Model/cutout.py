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

def apply_black_patch_to_bbox(image, bbox):
    h, w = image.shape[:2]
    class_label, x_center, y_center, width, height = bbox

    # Convert normalized values to actual pixel values
    x_center, y_center, width, height = int(x_center * w), int(y_center * h), int(width * w), int(height * h)
    x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
    x2, y2 = x1 + width, y1 + height

    # Calculate 10% area patch coordinates
    patch_width, patch_height = int(0.1 * width), int(0.1 * height)
    patch_x1, patch_y1 = random.randint(x1, x2 - patch_width), random.randint(y1, y2 - patch_height)
    patch_x2, patch_y2 = patch_x1 + patch_width, patch_y1 + patch_height

    # Fill the patch in the original image with black color
    image[patch_y1:patch_y2, patch_x1:patch_x2] = 0  # 0 corresponds to black in BGR

def save_annotations(annotations, file_path):
    with open(file_path, 'w') as file:
        for annotation in annotations:
            file.write(' '.join(map(str, annotation)) + '\n')


def main():
    image_dir = "/home/ch/Downloads/yolov5Model/data/images/train"
    label_dir = "/home/ch/Downloads/yolov5Model/data/labels/train"

    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    labels = [os.path.join(label_dir, os.path.splitext(os.path.basename(f))[0] + '.txt') for f in images]
    count = 0
    for i, image_path in enumerate(images):
        if count<20:
            count+=1
            label_path = labels[i]
            image, bboxes = load_image_and_bboxes(image_path, label_path)

            for bbox in bboxes:
                apply_black_patch_to_bbox(image, bbox)

            cv2.imwrite(f'/home/ch/Downloads/yolov5Model/data/images/train/cut_{i}.jpg', image)

            save_annotations(bboxes, f'/home/ch/Downloads/yolov5Model/data/labels/train/cut_{i}.txt')
        else:
            break
if __name__ == "__main__":
    main()
