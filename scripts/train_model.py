import os
from src.config_reader import config
from src.data_processing import load_annotations, extract_hog_features
from src.model import (
    create_knn_model,
    train_model,
    save_model,
    knn_hyperparameter_tuning,
    random_forest_hyperparameter_tuning
)


def main():
    data_path = config['data']['path']
    train_images_dir = config['data']['train_images_dir']
    train_labels_dir = config['data']['train_labels_dir']

    train_images_path = os.path.join(data_path, train_images_dir)
    train_labels_path = os.path.join(data_path, train_labels_dir)
    
    train_images, train_anotations = load_annotations(train_images_path, train_labels_path)
    # Trích xuất HOG features từ tập train
    resize_width, resize_height = 32, 64
    train_hog_features, train_labels = extract_hog_features(
        train_images, train_anotations, resize_width, resize_height)
    
    # Tìm kiếm siêu tham số tốt nhất và huấn luyện mô hình kNN
    best_knn = knn_hyperparameter_tuning(train_hog_features, train_labels)
    best_knn.fit(train_hog_features, train_labels)
    save_model(best_knn, os.path.join(
        config['output']['model_dir'], 'best_knn_model.pkl'))

    # Fine-tuning và huấn luyện mô hình Random Forest
    best_rf = random_forest_hyperparameter_tuning(
        train_hog_features, train_labels)
    best_rf.fit(train_hog_features, train_labels)
    save_model(best_rf, os.path.join(
        config['output']['model_dir'], 'best_rf_model.pkl'))


if __name__ == "__main__":
    main()
