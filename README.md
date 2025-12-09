# 🤟 American Sign Language (ASL) Detection

## 📌 Project Overview
This project is an **American Sign Language (ASL) Detection System** capable of recognizing hand gestures for the 26 letters of the alphabet (A-Z) and 3 special characters (Space, Delete, Nothing). It utilizes **Transfer Learning** with the **MobileNetV2** architecture to achieve high accuracy with efficient training. The project includes a training script to build the model and a **Streamlit** web application for real-time inference.

## 📂 Dataset Info
- **Source**: Kaggle ASL Alphabet Dataset.
- **Classes**: 29 Total Classes.
  - **Alphabets**: A-Z (26 classes)
  - **Special**: SPACE, DELETE, NOTHING (3 classes)
- **Structure**: The dataset is organized into separate folders for each class within the training directory.
- **Preprocessing**: Images are resized to **128x128** pixels and normalized (rescaled by 1/255).

## ⚙️ Methodology
The system uses **Transfer Learning** to leverage a pre-trained model for feature extraction:
1.  **Base Model**: **MobileNetV2** (pre-trained on ImageNet), with top layers excluded. The base layers are **frozen** to retain learned features.
2.  **Custom Head**:
    - `GlobalAveragePooling2D` layer.
    - `Dense` layer with **256 units** and **ReLU** activation.
    - `Dropout` layer (0.2) to prevent overfitting.
    - Final `Dense` output layer with **29 units** and **Softmax** activation for classification.
3.  **Data Augmentation**: Applied during training (Rotation, Zoom, Width/Height shift, Horizontal flip) to improve robustness.

## 🛠 Tech Stack
- **Language**: Python 3.x
- **Deep Learning**: TensorFlow, Keras
- **Web Framework**: Streamlit
- **Image Processing**: OpenCV, PIL (Pillow), NumPy
- **Utilities**: OS, JSON

## 📥 Installation Steps
1.  **Clone the repository**:
    ```
    git clone https://github.com/aiml-developer/ASL-Detection
    cd ASL-Detection
    ```

2.  **Install Dependencies**:
    Make sure you have Python installed. Run the following command:
    ```
    pip install tensorflow streamlit pillow numpy
    ```

3.  **Dataset Setup**:
    - Download the ASL dataset.
    - Place the training data in: `archive/asl_alphabet_train/` (or update paths in `train.py`).

## 🚀 Usage
### Running the Web App
To use the interface for detection:
1.  Ensure the trained model (`asl_detection_model.h5`) and indices (`class_indices.pkl`) are present in the `model/` directory.
2.  Run the Streamlit app:
    ```
    streamlit run app.py
    ```
3.  Upload an image of a hand sign to get the prediction and confidence score.

## 🧠 Model Training Steps
To retrain the model from scratch:
1.  Verify your dataset is in `archive/asl_alphabet_train`.
2.  Run the training script:
    ```
    python train.py
    ```
3.  **Process**:
    - The script loads MobileNetV2.
    - Trains for **5 Epochs** (default) using the Adam optimizer (`lr=0.001`).
    - Saves the trained model to `model/asl_detection_model.h5`.
    - Saves class mappings to `model/class_indices.json`.

## 📊 Results
- **Model Architecture**: MobileNetV2 (Lightweight & Fast).
- **Training Epochs**: 5.
- **Performance**: High accuracy is achieved quickly due to transfer learning on the diverse ImageNet features.
- **Output**: Predicts the class label with a percentage confidence score.

## 🎥 Demo Section
The **Streamlit UI** provides a simple interface:
1.  **Upload**: User uploads a `.jpg` or `.png` image.
2.  **Display**: The app shows the uploaded image.
3.  **Analyze**: Clicking "Analyze Sign" passes the image through the model.
4.  **Result**: Displays the predicted character (e.g., "**A**") and the confidence level (e.g., "98.5%").

## 📝 Conclusion
This project demonstrates an effective application of **Computer Vision** and **Deep Learning** for accessibility technology. By using MobileNetV2, the model remains lightweight enough for deployment on edge devices or web interfaces while maintaining high precision in recognizing ASL gestures.
