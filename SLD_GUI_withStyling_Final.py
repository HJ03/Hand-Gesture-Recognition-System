import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Button, Label, ttk
from PIL import Image, ImageTk
import pickle

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Code")
        self.root.configure(bg="#f0f0f0")  # Background color for aesthetics

        # **Styling setup**
        self.style = ttk.Style()

        # Title label
        self.style.configure("Title.TLabel", font=("Cambria", 60, "bold"), foreground="Black", background="#f0f0f0")
        self.title_label = ttk.Label(root, text="Vision Code", style="Title.TLabel")
        self.title_label.pack(pady=10)

        # Create a frame with a 2px border and curved corners for the video
        self.video_canvas = tk.Canvas(root, bd=2, highlightthickness=2, highlightbackground="black")
        self.video_canvas.pack(pady=20)

        # Label for the video display (set desired size)
        self.video_frame = Label(self.video_canvas, width=640, height=480)
        self.video_frame.pack(pady=10, padx=10)

        # Button styling: 
        self.close_button = tk.Button(
            root,
            text="CLOSE",
            font=("Arial", 16),
            bg="#007bff",          # Bootstrap primary button color
            fg="white",            # White text
            activebackground="#0056b3",  # Darker blue for hover effect
            activeforeground="white",
            relief="flat",         # No border
            bd=0,                  # Border width
            padx=20,               # Horizontal padding
            pady=10,               # Vertical padding
        )
        self.close_button.configure(cursor="hand2")  # Change to pointer when hovered
        self.close_button.pack(pady=10)

        # Attach command
        self.close_button.config(command=self.close_window)


        # Load the trained model
        with open('./model.p', 'rb') as f:
            model_dict = pickle.load(f)
        self.model = model_dict['model']

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'DELETE', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 
                            9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 
                            18: 'R', 19: 'S', 20: 'SPACE', 21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 
                            26: 'Y', 27: 'Z'}

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # Resize the frame to 640x480
        frame = cv2.resize(frame, (640, 480))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_, y_, data_aux = [], [], []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]

                x1, y1 = int(min(x_) * frame.shape[1]), int(min(y_) * frame.shape[0])
                x2, y2 = int(max(x_) * frame.shape[1]), int(max(y_) * frame.shape[0])
                cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (0, 0, 255), 2)
                cv2.putText(frame, predicted_character, (x1 - 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def close_window(self):
        self.cap.release()
        self.hands.close()
        self.root.destroy()



# Create the Tkinter window
root = tk.Tk()
app = HandGestureApp(root)
root.mainloop()
