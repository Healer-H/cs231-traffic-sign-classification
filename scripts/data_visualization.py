import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.config_reader import config
from src.data_processing import load_annotations
from src.visualization import (
    plot_cropped_images_by_class,
    visualize_hog_features,
    plot_data_distribution,
    visualize_bboxes,
    count_labels
)

def main():
    data_dir = config['data']['path']
    train_images_dir = os.path.join(data_dir, config['data']['train_images_dir'])
    train_labels_dir = os.path.join(data_dir, config['data']['train_labels_dir'])
    test_images_dir = os.path.join(data_dir, config['data']['test_images_dir'])
    test_labels_dir = os.path.join(data_dir, config['data']['test_labels_dir'])
    valid_images_dir = os.path.join(data_dir, config['data']['valid_images_dir'])
    valid_labels_dir = os.path.join(data_dir, config['data']['valid_labels_dir'])
    
    # Tạo thư mục để lưu trữ hình ảnh trực quan hóa
    visualizations_dir = os.path.join(config['output']['results_dir'], 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)

    # Load annotations từ tập train và test
    train_images, train_annotations = load_annotations(train_images_dir, train_labels_dir)
    test_images, test_annotations = load_annotations(test_images_dir, test_labels_dir)
    valid_images, valid_annotations = load_annotations(valid_images_dir, valid_labels_dir)
    
    test_images = test_images + valid_images
    test_annotations = test_annotations + valid_annotations

    # Trực quan hóa bounding boxes
    visualize_bboxes(train_images, train_annotations, visualizations_dir, num_images=5)

    train_counts = count_labels(train_annotations)
    test_counts = count_labels(test_annotations)

    plot_data_distribution(train_counts, test_counts, visualizations_dir)

if __name__ == "__main__":
    main()