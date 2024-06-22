import os
import logging
import numpy as np
from src.config_reader import config
from data_processing import load_annotations, extract_hog_features
from model import load_model, evaluate_model

def main():
    logging.basicConfig(level=logging.INFO)

    data_path = config['data']['path']
    test_images_dir = config['data']['test_images_dir']
    test_labels_dir = config['data']['test_labels_dir']
    
    valid_images_dir = config['data']['valid_images_dir']
    valid_labels_dir = config['data']['valid_labels_dir']

    test_images_path = os.path.join(data_path, test_images_dir)
    test_labels_path = os.path.join(data_path, test_labels_dir)
    
    valid_images_path = os.path.join(data_path, valid_images_dir)
    valid_labels_path = os.path.join(data_path, valid_labels_dir)
    
    test_images, test_labels = load_annotations(test_images_path, test_labels_path)
    valid_images, valid_labels = load_annotations(valid_images_path, valid_labels_path)
    resize_width, resize_height = 32, 64
    test_hog_features, test_labels = extract_hog_features(test_images, test_labels, resize_width, resize_height)
    valid_hog_features, valid_labels = extract_hog_features(valid_images, valid_labels, resize_width, resize_height)

    # concatenate test and valid data
    test_hog_features = np.concatenate((test_hog_features, valid_hog_features), axis=0)
    test_labels = np.concatenate((test_labels, valid_labels), axis=0)
    
    
    # Đánh giá mô hình kNN
    knn_model_path = os.path.join(
        config['output']['model_dir'], 
        config['output']['knn_model_dir'], 
        config['output']['knn_model_file']
    )
    knn_model = load_model(knn_model_path)
    evaluate_model(knn_model, test_hog_features, test_labels, 'kNN')

    # Đánh giá mô hình Random Forest
    rf_model_path = os.path.join(
        config['output']['model_dir'], 
        config['output']['random_forest_model_dir'],
        config['output']['random_forest_model_file'])
    rf_model = load_model(rf_model_path)
    evaluate_model(rf_model, test_hog_features, test_labels, 'Random Forest')

if __name__ == "__main__":
    main()