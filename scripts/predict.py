import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
from skimage.feature import hog
from src.model import load_model

random_forest_model = load_model('models/random_forest/best_rf_model.pkl')

knn_model = load_model('models/knn/best_knn_model.pkl')

def preprocess_image(image, resize_width=32, resize_height=64):
    resized_image = cv2.resize(image, (resize_width, resize_height))
    
    hog_channels = []
    for channel in range(resized_image.shape[2]):
        fd = hog(resized_image[:, :, channel], pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_channels.append(fd)
    
    hog_feature = np.concatenate(hog_channels)

    reshaped_features = hog_feature.reshape(1, -1)
    return reshaped_features

def predict_image(model, image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction[0]

def load_image():
    global img_path, img
    img_path = filedialog.askopenfilename()
    if img_path:
        img = cv2.imread(img_path)
        display_image(img)

def display_image(img):
    max_size = 400 
    height, width = img.shape[:2]
    scale = min(max_size / width, max_size / height)
    if scale < 1:
        img = cv2.resize(img, (int(width * scale), int(height * scale)))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    panel1.configure(image=img_tk)
    panel1.image = img_tk
    panel2.configure(image=img_tk)
    panel2.image = img_tk

def predict_and_display():
    if not img_path:
        return
    
    predicted_label_rf = predict_image(random_forest_model, img)
    
    predicted_label_knn = predict_image(knn_model, img)
    
    img_with_label_rf = img.copy()
    img_with_label_knn = img.copy()
    
    colors = [(10,10,240), (15,215,255), (170,70,0), (212,162,127)]
    labels = ['Prohibition', 'Warning', 'Mandatory', 'Information']
    cv2.putText(img_with_label_rf, labels[predicted_label_rf], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[predicted_label_rf], 2)
    cv2.putText(img_with_label_knn, labels[predicted_label_knn], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[predicted_label_knn], 2)
    
    img_rgb_rf = cv2.cvtColor(img_with_label_rf, cv2.COLOR_BGR2RGB)
    img_pil_rf = Image.fromarray(img_rgb_rf)
    img_tk_rf = ImageTk.PhotoImage(img_pil_rf)
    panel1.configure(image=img_tk_rf)
    panel1.image = img_tk_rf

    img_rgb_knn = cv2.cvtColor(img_with_label_knn, cv2.COLOR_BGR2RGB)
    img_pil_knn = Image.fromarray(img_rgb_knn)
    img_tk_knn = ImageTk.PhotoImage(img_pil_knn)
    panel2.configure(image=img_tk_knn)
    panel2.image = img_tk_knn

    label_rf.config(text=f"Random Forest: {labels[predicted_label_rf]}")
    label_knn.config(text=f"kNN: {labels[predicted_label_knn]}")

root = tk.Tk()
root.title("Traffic Sign Detection")

btn_load_image = tk.Button(root, text="Load Image", command=load_image)
btn_load_image.pack()

btn_predict = tk.Button(root, text="Predict and Display", command=predict_and_display)
btn_predict.pack()

panel1 = tk.Label(root)
panel1.pack(side="left", padx=10, pady=10)

panel2 = tk.Label(root)
panel2.pack(side="right", padx=10, pady=10)

label_rf = tk.Label(root, text="Random Forest:")
label_rf.pack(side="left", padx=10, pady=10)

label_knn = tk.Label(root, text="kNN:")
label_knn.pack(side="right", padx=10, pady=10)

img_path = None
img = None

root.mainloop()