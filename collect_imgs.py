import os
import cv2

# Directory where data will be stored
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

# Number of classes and dataset size
number_of_classes = 28
dataset_size = 100

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use index 0 for the default camera
if not cap.isOpened():
    print("Error: Could not open the video capture.")
    exit()

# Loop through each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f'Collecting data for class {j}')

    # Display prompt until user is ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame for a mirror view
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Start collecting frames
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame for a mirror view
        frame = cv2.flip(frame, 1)

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save frame to the respective class directory
        frame_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(frame_path, frame)
        counter += 1

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
