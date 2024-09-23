# -*- coding: utf-8 -*-
"""car_num_plate-detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JRstP3br48KzoUog2mpy_ZkpOw5dof9r
"""

pip install tensorflow keras opencv-python matplotlib numpy pandas scikit-learn albumentations

import pandas as pd

# Paths to annotation files
train_set_1_annotations = '/content/Licplatesdetection_train.csv'
train_set_2_annotations = '/content/Licplatesrecognition_train.csv'
# test_set_annotations = 'dataset/test_set/annotations.csv'

# Load annotations
df_train1 = pd.read_csv(train_set_1_annotations)
df_train2 = pd.read_csv(train_set_2_annotations)
# df_test = pd.read_csv(test_set_annotations)

print("Training Set 1 Annotations:", df_train1.head())
print("Training Set 2 Annotations:", df_train2.head())
# print("Test Set Annotations:", df_test.head())

print(df_train1['img_id'].head(1))

import os

for img_path in df_train1['img_id']:
    if not os.path.exists(img_path):
        print(f"File does not exist: {img_path}")

for i in range(1):
    img_path = df_train1.iloc[i]['img_id']
    print(f"Trying to load: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not load image {img_path}")
        continue

import matplotlib.pyplot as plt
import cv2

def visualize_bounding_boxes(df, num_samples=5):
    for i in range(num_samples):
        img_path = df.iloc[i]['img_id']
        image = cv2.imread(img_path)

        # Check if the image was loaded correctly
        if image is None:
            print(f"Error: Could not load image {img_path}")
            continue  # Skip to the next image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ymin = df.iloc[i]['ymin']
        xmin = df.iloc[i]['xmin']
        ymax = df.iloc[i]['ymax']
        xmax = df.iloc[i]['xmax']

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        plt.imshow(image)
        plt.title(f"Image {i+1} with Bounding Box")
        plt.axis('off')
        plt.show()

# Visualize bounding boxes from Training Set 1
visualize_bounding_boxes(df_train1, num_samples=3)



from collections import Counter

# Display sample texts
print("Sample License Plate Texts:", df_train2['text'].head())

# Character frequency
all_text = ''.join(df_train2['text'].astype(str).values)
char_counts = Counter(all_text)
print("Character Frequencies:", char_counts)

import seaborn as sns

# Calculate width and height of bounding boxes
df_train1['width'] = df_train1['xmax'] - df_train1['xmin']
df_train1['height'] = df_train1['ymax'] - df_train1['ymin']

# Plot distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_train1['width'], bins=30, kde=True)
plt.title('Distribution of Bounding Box Widths')

plt.subplot(1, 2, 2)
sns.histplot(df_train1['height'], bins=30, kde=True)
plt.title('Distribution of Bounding Box Heights')

plt.show()

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5.git
# %cd yolov5
!pip install -r requirements.txt

import yaml

data = {
    'train': '/content/drive/MyDrive/Licplatesdetection_train/license_plates_detection_train',
    'val': '/content/drive/MyDrive/test/test/test',  # Ideally, split a portion for validation
    'nc': 1,  # number of classes
    'names': ['license_plate']
}

# Save to dataset.yaml
# with open('dataset.yaml', 'w') as file:
#     yaml.dump(data, file)
with open('dataset.yaml', 'r') as file:
    config = yaml.safe_load(file)
    print(config)

import os

def convert_to_yolo(df, images_dir, labels_dir):
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    for idx, row in df.iterrows():
        img_path = row['img_id']
        img_name = os.path.basename(img_path).split('.')[0]
        txt_path = os.path.join(labels_dir, f"{img_name}.txt")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not load image {img_path}")
            continue

        height, width, _ = image.shape
        # YOLO format: class x_center y_center width height (all normalized)
        class_id = 0  # 'license_plate' class

        x_center = ((row['xmin'] + row['xmax']) / 2) / width
        y_center = ((row['ymin'] + row['ymax']) / 2) / height
        bbox_width = (row['xmax'] - row['xmin']) / width
        bbox_height = (row['ymax'] - row['ymin']) / height

        with open(txt_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

# Define directories
images_dir = '/content/drive/MyDrive/Licplatesdetection_train/license_plates_detection_train'
labels_dir = '/content/drive/MyDrive/Licplatesrecognition_train/license_plates_recognition_train'

# Convert annotations
convert_to_yolo(df_train1, images_dir, labels_dir)

# Commented out IPython magic to ensure Python compatibility.
# Navigate to the YOLOv5 directory
# %cd /content/yolov5

# Train the model
!python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name license_plate_detector

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import string

# Define characters (assuming uppercase letters and digits)
characters = string.ascii_uppercase + string.digits
num_classes = len(characters) + 1  # +1 for the CTC blank label

# Create a character to index mapping
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}  # 0 is reserved for blank

def encode_text(text):
    return [char_to_num[char] for char in text if char in char_to_num]

df_train2['encoded'] = df_train2['text'].apply(encode_text)

from tensorflow.keras import layers, models

def build_crnn_model(input_shape, num_classes):
    input_img = layers.Input(shape=input_shape, name='input_img')

    # Convolutional layers
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)

    # Flatten the output before passing to RNN
    x = layers.Reshape((-1, x.shape[-1]))(x)  # Reshape to (timesteps, features)

    # Recurrent layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Output layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_img, outputs=x)
    return model

# Define input shape (height, width, channels)
input_shape = (64, 256, 1)  # Adjust as needed
num_classes = 10  # Define the number of classes
model_crnn = build_crnn_model(input_shape, num_classes)
model_crnn.summary()



!pip install easyocr

import cv2
import os
import numpy as np
from google.colab.patches import cv2_imshow
import easyocr  # Efficient for OCR tasks

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blur, 30, 150)

    # Find contours and filter based on size and aspect ratio
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    license_plate_candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        if 2 < aspect_ratio < 5 and area > 500:  # Modify thresholds based on the expected plate size and aspect ratio
            license_plate_candidates.append((x, y, w, h))

    return license_plate_candidates, image

def extract_license_plate_text(image_region):
    results = reader.readtext(image_region, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return ' '.join([text for _, text, _ in results])

def process_test_images(directory_path):
    detected_license_plates = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(directory_path, filename)
            candidates, image = preprocess_image(image_path)

            for (x, y, w, h) in candidates:
                license_plate_region = image[y:y+h, x:x+w]
                cv2_imshow(license_plate_region)  # Show detected plate area
                plate_text = extract_license_plate_text(license_plate_region)
                print(f"Detected License Plate Text: {plate_text}")
                detected_license_plates.append({'image': filename, 'license_plate_text': plate_text})

    return detected_license_plates

# Path to the directory containing the test images
test_images_directory = '/content/drive/MyDrive/test/test/test'

# Process all test images
results = process_test_images(test_images_directory)

# Output results
for result in results:
    print(f"Image: {result['image']}, Detected License Plate: {result['license_plate_text']}")

