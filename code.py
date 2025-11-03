import os
import matplotlib.pyplot as plt
import numpy as np
import random, shutil
import cv2
from pygame import mixer  # For playing the alarm sound
import time

# --- All imports updated to tensorflow.keras ---
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization

# -----------------------------------------------------------------
# PART 1: MODEL TRAINING
# -----------------------------------------------------------------

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    """Creates a data generator. 'Closed' will be 0, 'Open' will be 1."""
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale',
        class_mode=class_mode,
        target_size=target_size
    )

# --- 1. Setup Data Generators ---
BS = 32  # Batch Size
TS = (24, 24)  # Target Size
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)
SPE = len(train_batch.classes) // BS  # Steps Per Epoch
VS = len(valid_batch.classes) // BS   # Validation Steps
print(f"Steps per Epoch: {SPE}, Validation Steps: {VS}")

# --- 2. Build the Model ---
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: 0=Closed, 1=Open
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()  # Prints a summary of the model layers

# --- 3. Train the Model ---
print("Starting model training...")
model.fit(
    train_batch,
    validation_data=valid_batch,
    epochs=15,
    steps_per_epoch=SPE,
    validation_steps=VS
)
print("Model training finished.")

model.save('models/drowsiness_model.keras', overwrite=True)
print("Model saved as 'models/drowsiness_model.keras'")

# -----------------------------------------------------------------
# PART 2: REAL-TIME DETECTION
# -----------------------------------------------------------------

# --- 1. Initialize Sound and Classifiers ---
mixer.init()
sound = mixer.Sound('alarm.wav')

face_cascade_path = r'haar cascade files\haarcascade_frontalface_alt.xml'
left_eye_cascade_path = r'haar cascade files\haarcascade_lefteye_2splits.xml'
right_eye_cascade_path = r'haar cascade files\haarcascade_righteye_2splits.xml'

face = cv2.CascadeClassifier(face_cascade_path)
leye = cv2.CascadeClassifier(left_eye_cascade_path)
reye = cv2.CascadeClassifier(right_eye_cascade_path)

if face.empty():
    print(f"FATAL ERROR: Could not load face cascade from {face_cascade_path}")
    exit()
if leye.empty():
    print(f"FATAL ERROR: Could not load left eye cascade from {left_eye_cascade_path}")
    exit()
if reye.empty():
    print(f"FATAL ERROR: Could not load right eye cascade from {right_eye_cascade_path}")
    exit()

print("Haar cascades loaded successfully.")

model = load_model('models/drowsiness_model.keras')

path = os.getcwd()
cap = cv2.VideoCapture(0)  # Start webcam
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
alarm_on = False  # --- ADDED: Alarm state
print("Starting detection... Press 'q' to quit.")

# --- 2. Start Detection Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam. Exiting...")
        break
        
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- FINAL LOGIC FIX 1 ---
    # Default prediction is 'Open' (1)
    # This prevents score from increasing if eyes aren't detected
    rpred = [1]
    lpred = [1]

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255  # Normalize
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=1)
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255  # Normalize
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=1)
        break

    # --- 3. Drowsiness Logic (FINAL FIX 2) ---
    
    # Check if EITHER eye is predicted as 'Closed' (0)
    if(rpred[0] == 0 or lpred[0] == 0):
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:  # Both eyes are 'Open' (1)
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
    if score < 0:
        score = 0
        
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Optional: Print debug info
    # print(f"Left: {lpred[0]}, Right: {rpred[0]}, Score: {score}")

    # --- 4. Trigger Alarm (FINAL LOGIC) ---
    # INCREASE THIS NUMBER (e.g., 30 or 40) to make it less sensitive
    if score > 30: 
        if not alarm_on:
            try:
                sound.play(-1)  # The -1 makes the sound LOOP
                alarm_on = True
            except:
                pass
        
        # Flashing red border
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    
    else:
        # If score is low, turn off the alarm
        if alarm_on:
            sound.stop()
            alarm_on = False

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Cleanup ---
cap.release()
sound.stop() # Make sure sound is off on exit
cv2.destroyAllWindows()
print("Application closed successfully.")
