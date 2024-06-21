import os
from src.config_reader import config
from src.data_processing import load_annotations, extract_hog_features
from src.model import (
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

    train_images, train_anotations = load_annotations(
        train_images_path, train_labels_path)

    resize_width, resize_height = 32, 64
    train_hog_features, train_labels = extract_hog_features(
        train_images, train_anotations, resize_width, resize_height)

    best_knn = knn_hyperparameter_tuning(train_hog_features, train_labels)
    print("Training the best kNN model...")
    best_knn.fit(train_hog_features, train_labels)
    save_model(best_knn, os.path.join(
        config['output']['model_dir'],
        config['output']['knn_model_dir'],
        'best_knn_model.pkl')
    )

    best_rf = random_forest_hyperparameter_tuning(
        train_hog_features, train_labels)
    print("Training the best Random Forest model...")
    best_rf.fit(train_hog_features, train_labels)
    save_model(best_rf, os.path.join(
        config['output']['model_dir'],
        config['output']['random_forest_model_dir'],
        'best_rf_model.pkl')
    )


if __name__ == "__main__":
    main()
