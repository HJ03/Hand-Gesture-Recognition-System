import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Using the Hands model in static mode for image processing
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing image data
DATA_DIR = './data'

# Lists to hold data and labels
data = []
labels = []

# Loop through each class directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Skip if not a directory

    # Loop through each image in the directory
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)

        if img is None:
            print(f"Warning: Could not read image {img_full_path}")
            continue

        data_aux = []
        x_ = []
        y_ = []

        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image using MediaPipe Hands
        results = hands.process(img_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize landmarks relative to the minimum x and y
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(int(dir_))  # Convert label to integer if directories are numeric

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Release MediaPipe resources
hands.close()

print("Data collection and serialization complete.")
