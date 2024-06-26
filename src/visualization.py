import matplotlib.pyplot as plt
import pandas as pd
import cv2
import random
import numpy as np
from skimage.feature import hog
from config_reader import config
import bbox_visualizer as bbv
import os 

def plot_cropped_images_by_class(images, annotations, class_label, num_images=30):
    cropped_images = []

    for img, (bboxes, labels) in zip(images, annotations):
        for bbox, label in zip(bboxes, labels):
            if label == class_label:
                x1, y1, x2, y2 = bbox
                cropped_img = img[y1:y2, x1:x2]
                cropped_img = cv2.resize(cropped_img, (64, 64))
                cropped_images.append(cropped_img)

    random.shuffle(cropped_images)

    cropped_images = cropped_images[:num_images]

    _, axes = plt.subplots(3, 10, figsize=(15, 5))
    for ax, img in zip(axes.flatten(), cropped_images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    plt.show()


def visualize_hog_features(images, annotations, class_label, num_images=5, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    for img, (bboxes, labels) in zip(images, annotations):
        for bbox, label in zip(bboxes, labels):
            if label == class_label:
                x1, y1, x2, y2 = bbox
                cropped_img = img[y1:y2, x1:x2]
                resize_width = config['images']['resize_width']
                resize_height = config['images']['resize_height']
                resized_img = cv2.resize(cropped_img, (resize_width, resize_height))

                # Tính toán HOG cho từng kênh màu và kết hợp các kết quả
                hog_features = []
                hog_images = []
                for channel in range(resized_img.shape[2]):
                    fd, hog_image = hog(
                        resized_img[:, :, channel], 
                        pixels_per_cell=pixels_per_cell, 
                        cells_per_block=cells_per_block, 
                        visualize=True
                        )
                    hog_features.append(fd)
                    hog_images.append(hog_image)

                hog_features = np.concatenate(hog_features)
                _ = np.sum(hog_images, axis=0)

                _, ax = plt.subplots(1, 2, figsize=(5, 5))
                ax[0].imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
                ax[0].set_title('Original Image')
                ax[0].axis('off')

                ax[1].imshow(hog_images[0], cmap='gray')
                ax[1].set_title('HOG Features')
                ax[1].axis('off')

                plt.show()

                num_images -= 1
                if num_images == 0:
                    return

def visualize_bboxes(images, annotations, output_dir, num_images=5):
    colors = [
        (240, 10, 10),  # Red
        (255, 215, 15), # Yellow
        (0, 70, 170),   # Blue
        (127, 162, 212) # Light Blue
    ]
    label_names = ['P', 'W', 'M', 'I']
    cropped_images = []
    
    for i in range(min(num_images, len(images))):
        img = images[i].copy()
        bboxes, labels = annotations[i]

        for bbox, label in zip(bboxes, labels):
            color = colors[label]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_name = label_names[label]
            cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'image_{i}.png'))
        plt.close()
        cropped_images.append(img)
    
    return cropped_images

def count_labels(annotations):
    label_counts = [0, 0, 0, 0]
    for bboxes, labels in annotations:
        for label in labels:
            label_counts[label] += 1
    return label_counts

def plot_data_distribution(train_counts, test_counts, output_dir):
    data = {
        'Class': ['Prohibition', 'Warning', 'Mandatory', 'Information'],
        'Train': train_counts,
        'Test': test_counts,
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    ax = df.set_index('Class').T.plot(kind='bar', figsize=(12, 6), legend=True, color=['#F00A0A', '#FFD70F', '#0046AA', '#7FA2D4'])
    plt.title('Data Distribution')
    plt.ylabel('Count')
    plt.xlabel('Dataset')
    plt.xticks(rotation=45)
    
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), color='black', fontweight='bold', ha='center', va='center', fontsize=11, xytext=(0, 10),
                    textcoords='offset points')
    
    plt.legend(title='Class', labels=['P', 'W', 'M', 'I'])
    plt.savefig(os.path.join(output_dir, 'data_distribution.png'))
    plt.close()