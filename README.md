✋#Sign Language Alphabet Detection using Machine Learning#
A machine learning-based system designed to detect and classify sign language alphabet gestures in real time using image input. This project applies a structured three-phase methodology including dataset creation, model training, and real-time inference.

📌 Features
Real-time hand gesture recognition

Alphabet classification using Random Forest

Feature extraction from gesture images

Custom dataset support

Live camera input for prediction and feedback

🧠 Methodology
The system development is divided into three main stages:

1. Dataset Creation and Preprocessing
A custom dataset of hand gesture images is created and preprocessed to extract relevant features for model training.

Image Loading & Feature Extraction: Images are loaded and essential hand landmarks are extracted.

Normalization & Labeling: Features are normalized and assigned corresponding gesture labels.

Storage: Processed features and labels are saved for efficient reuse in training.

2. Model Training
The dataset is used to train a machine learning model using the Random Forest algorithm.

Train-Test Split: Data is divided into training and test sets with a balanced representation of each gesture.

Model Training: The Random Forest model is trained on extracted features.

Evaluation: Model performance is measured using accuracy metrics and classification reports.

Model Saving: The trained model is saved for use in real-time classification.

3. Real-Time Inference and Classification
The trained model is deployed for live hand gesture recognition through camera input.

Live Image Capture: Images are captured from a webcam or video feed.

Feature Extraction: Features from the live feed are extracted and normalized.

Prediction: The model classifies the gesture and displays results instantly.

Visual Feedback: Recognized gestures are shown on-screen with feature visualization.

🛠️ Technologies Used
Python

OpenCV

Scikit-learn

MediaPipe (for hand landmark detection)

Random Forest (classification algorithm)
