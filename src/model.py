from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle
import os
from src.config_reader import config


def create_knn_model(n_neighbors=3):
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def train_model(model, X_train, y_train):
    print(f"Ready to train model with {model.__class__.__name__}")
    model.fit(X_train, y_train)
    print(f"Model {model.__class__.__name__} has been trained successfully!")
    return model


def save_model(model, model_path):
    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)


def load_model(model_path):
    print(f"Loading model from {model_path}")
    return joblib.load(model_path)


def knn_hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_neighbors': config['model']['knn']['n_neighbors'],
        'weights': config['model']['knn']['weights'],
    }
    print("Start tuning hyperparameters for kNN model")
    print(f"Parameters grid: {param_grid}")

    knn = KNeighborsClassifier()

    grid_search_knn = GridSearchCV(
        knn, param_grid, cv=5, scoring='accuracy', return_train_score=True)
    grid_search_knn.fit(X_train, y_train)
    
    print("Hyperparameter tuning for kNN model has been done!")
    print(f"Best hyperparameters: {grid_search_knn.best_params_}")
    print(f"Saving results to file {config['output']['knn_results_file']}")

    knn_results_df = pd.DataFrame(grid_search_knn.cv_results_)
    output_dir_path = os.path.join(
    config['output']['results_dir'], 
    config['output']['hyperparameter_tuning_dir']
    )
    os.makedirs(output_dir_path, exist_ok=True)  # Create the directory if it doesn't exist

    knn_results_df.to_csv(os.path.join(
        output_dir_path, 
        config['output']['knn_results_file']), index=False)

    return grid_search_knn.best_estimator_


def random_forest_hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': config['model']['random_forest']['n_estimators'],
        'max_depth': config['model']['random_forest']['max_depth']
    }
    print("Start tuning hyperparameters for Random Forest model")
    print(f"Parameters grid: {param_grid}")
    
    rf = RandomForestClassifier()

    grid_search_rf = GridSearchCV(rf, param_grid, cv=5,
                                  scoring='accuracy', return_train_score=True)
    grid_search_rf.fit(X_train, y_train)

    print("Hyperparameter tuning for Random Forest model has been done!")
    print(f"Best hyperparameters: {grid_search_rf.best_params_}")
    print(f"Saving results to file {config['output']['random_forest_results_file']}")
    
    output_dir_path = os.path.join(
        config['output']['results_dir'], 
        config['output']['hyperparameter_tuning_dir']
        )
    os.makedirs(output_dir_path, exist_ok=True)  # Create the directory if it doesn't exist

    
    rf_results_df = pd.DataFrame(grid_search_rf.cv_results_)
    rf_results_df.to_csv(os.path.join(
        output_dir_path,
        config['output']['random_forest_results_file']), index=False)

    return grid_search_rf.best_estimator_


def read_knn_results(file_path):
    print(f"Reading kNN results from {file_path}")
    knn_results_df = pd.read_csv(file_path)
    knn_results_df = knn_results_df.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                                                  'split0_train_score', 'split1_train_score', 'split2_train_score', 'split3_train_score', 'split4_train_score', 'std_train_score',
                                                  'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score',
                                                  'std_test_score']).sort_values(by=['rank_test_score'])
    return knn_results_df


def read_rf_results(file_path):
    print(f"Reading Random Forest results from {file_path}")
    rf_results_df = pd.read_csv(file_path)
    rf_results_df = rf_results_df.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                                                'split0_train_score', 'split1_train_score', 'split2_train_score', 'split3_train_score', 'split4_train_score', 'std_train_score',
                                                'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'std_test_score']).sort_values(by=['rank_test_score'])
    return rf_results_df
