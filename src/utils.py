import numpy as np
import pandas as pd
import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def load_image(image_path):
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_image(image, size=(64, 64)):
    import cv2
    return cv2.resize(image, size)

def normalize_images(images):
    return images / 255.0

def save_image(image, path):
    import cv2
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved image at: {path}")

def calculate_average_size(annotations):
    widths = []
    heights = []
    for bboxes, _ in annotations:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            widths.append(x2 - x1)
            heights.append(y2 - y1)
    
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    return avg_width, avg_height

def find_nearest_power_of_two(value):
    power = 1
    while power * 2 <= value:
        power *= 2
    return power

def count_labels(annotations):
    label_counts = [0, 0, 0, 0]
    for bboxes, labels in annotations:
        for label in labels:
            label_counts[label] += 1
    return label_counts
