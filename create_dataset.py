import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './train2'
NUM_LANDMARKS = 21  # Number of landmarks for a single hand

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        img_full_path = os.path.join(DATA_DIR, dir_, img_path)  # Full path to the image
        img = cv2.imread(img_full_path)

        # Check if the image was successfully loaded
        if img is None:
            print(f"Failed to load image: {img_full_path}")
            continue  # Skip to the next image

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalizing the landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Ensure the number of landmarks is the same for every image
                if len(data_aux) // 2 == NUM_LANDMARKS:  # 2 values (x, y) per landmark
                    data.append(data_aux)
                    labels.append(dir_)
                    print(f"Data appended for image: {img_full_path}")
                else:
                    print(f"Data not appended for image: {img_full_path}, number of landmarks: {len(data_aux) // 2}")

        else:
            print(f"No hand landmarks found in image: {img_full_path}")

# Saving the data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
