# 🧠 Sign Language Alphabet Detection using Machine Learning

A real-time sign language alphabet recognition system built with Python, MediaPipe, and a Random Forest classifier. The project is structured into three main phases: dataset creation and preprocessing, model training, and real-time inference using webcam input.

---

## 📌 Features

- Detects and classifies hand gestures representing alphabets.
- Real-time gesture recognition using webcam feed.
- Uses hand landmark detection for robust feature extraction.
- Custom dataset creation and preprocessing.
- Trained using the Random Forest classification algorithm.
- Visual feedback with prediction results displayed live.

---

## 🧠 Methodology

### 1. Dataset Creation and Preprocessing

- **Image Collection**: Hand gesture images are collected and organized into labeled directories by gesture.
- **Feature Extraction**: Uses MediaPipe to extract hand landmarks from images.
- **Normalization**: Features are normalized for consistency.
- **Labeling**: Each processed feature set is paired with the correct label.
- **Storage**: Processed features and labels are stored for training.

### 2. Model Training

- **Data Splitting**: Dataset is split into training and testing sets.
- **Training**: A Random Forest classifier is trained on the extracted features.
- **Evaluation**: Performance is measured using accuracy, precision, recall, and F1-score.
- **Model Saving**: The trained model is saved for use in real-time inference.

### 3. Real-Time Inference

- **Live Input**: Captures video from webcam.
- **Feature Extraction**: Processes live images to extract hand landmarks.
- **Prediction**: Classifies the gesture using the trained model.
- **Feedback**: Displays the predicted alphabet on the screen.

---


## 🛠️ Tech Stack

- **Programming Language**: Python
- **Libraries**: OpenCV, MediaPipe, Scikit-learn, NumPy, joblib
- **Model**: Random Forest Classifier

---

## 📊 Results

The system demonstrates:

- ✅ High accuracy in classifying alphabetic hand gestures
- ⚡ Fast real-time inference using webcam input
- 🔍 Reliable feature extraction with MediaPipe for landmark detection
- 📈 Consistent performance across various lighting conditions and hand positions
- 👁️ Visual feedback with recognized gesture displayed live on-screen

---




