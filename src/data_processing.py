import os
import cv2
import numpy as np
from skimage.feature import hog


def load_annotations(image_folder, label_folder):
    images = []
    annotations = []
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        if os.path.exists(label_path):
            img = cv2.imread(img_path)
            with open(label_path, 'r') as f:
                annots = f.readlines()

            bboxes = []
            labels = []
            for annot in annots:
                parts = annot.strip().split()
                if len(parts) == 0:
                    continue
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                x1 = int((x_center - width / 2) * img.shape[1])
                y1 = int((y_center - height / 2) * img.shape[0])
                x2 = int((x_center + width / 2) * img.shape[1])
                y2 = int((y_center + height / 2) * img.shape[0])

                bboxes.append([x1, y1, x2, y2])
                labels.append(label)

            images.append(img)
            annotations.append((bboxes, labels))

    return images, annotations


def extract_hog_features(images, annotations, resize_width, resize_height):
    hog_features = []
    labels = []

    for img, (bboxes, labels_list) in zip(images, annotations):
        for bbox, label in zip(bboxes, labels_list):
            x1, y1, x2, y2 = bbox
            cropped_img = img[y1:y2, x1:x2]
            resized_img = cv2.resize(
                cropped_img, (resize_width, resize_height))

            hog_channels = []
            for channel in range(resized_img.shape[2]):
                fd = hog(resized_img[:, :, channel], pixels_per_cell=(
                    8, 8), cells_per_block=(2, 2), visualize=False)
                hog_channels.append(fd)

            hog_feature = np.concatenate(hog_channels)
            hog_features.append(hog_feature)
            labels.append(label)

    return np.array(hog_features), np.array(labels)
