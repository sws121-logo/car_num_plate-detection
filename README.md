# README for Car Number Plate Detection Project

## Overview
This project implements an automatic car number plate detection and recognition system using deep learning techniques. The system utilizes YOLOv5 for object detection and a CRNN (Convolutional Recurrent Neural Network) model for license plate text recognition. The project is built on Google Colab for easy execution and deployment.

## Project Structure
- **Data**: Contains training and testing datasets in CSV format.
- **Models**: YOLOv5 for object detection and CRNN for text recognition.
- **Scripts**: Jupyter Notebook (`car_num_plate-detection.ipynb`) that contains all the code for data processing, model training, and evaluation.

## Requirements
To run this project, you need to install the following libraries:
```bash
pip install tensorflow keras opencv-python matplotlib numpy pandas scikit-learn albumentations easyocr
```

## Data Preparation
1. **Annotation Files**: 
   - `Licplatesdetection_train.csv`: Contains bounding box annotations for training the detection model.
   - `Licplatesrecognition_train.csv`: Contains text annotations for training the recognition model.

2. **Loading Annotations**:
   The annotations are loaded using pandas:
   ```python
   import pandas as pd
   df_train1 = pd.read_csv('path/to/Licplatesdetection_train.csv')
   df_train2 = pd.read_csv('path/to/Licplatesrecognition_train.csv')
   ```

3. **Visualizing Bounding Boxes**:
   The `visualize_bounding_boxes` function displays images with bounding boxes drawn around detected license plates.

## Model Training
### YOLOv5 for License Plate Detection
1. **Clone YOLOv5 Repository**:
   ```bash
   !git clone https://github.com/ultralytics/yolov5.git
   ```

2. **Install Requirements**:
   ```bash
   !pip install -r yolov5/requirements.txt
   ```

3. **Prepare Dataset Configuration**:
   Create a `dataset.yaml` file specifying paths to training and validation datasets.

4. **Train the Model**:
   Execute the training command:
   ```bash
   !python yolov5/train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --cfg yolov5/models/yolov5s.yaml --weights yolov5s.pt --name license_plate_detector
   ```

### CRNN for License Plate Recognition
1. **Character Encoding**:
   The license plate text is encoded into numerical format for model training.

2. **Model Architecture**:
   A CRNN model is built using Keras, consisting of convolutional layers followed by recurrent layers.

3. **Training the CRNN Model**:
   Compile and fit the model on the encoded text data.

## Testing
The system processes test images to detect and recognize license plates:
1. **Preprocess Images**: Convert images to grayscale, apply Gaussian blur, and perform edge detection.
2. **Extract License Plate Text**: Use EasyOCR to read text from the detected license plate regions.

## Running the Project
1. Upload your images to the specified directory.
2. Run the `process_test_images` function to detect and recognize license plates in the images.

## Results
The detected license plate text is printed for each image processed.

## Conclusion
This project demonstrates an effective approach to automatic license plate detection and recognition using state-of-the-art deep learning models. The combination of YOLOv5 and CRNN provides a robust solution for real-time applications in various domains, such as traffic monitoring and vehicle identification.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to modify any sections to suit your project specifics!
