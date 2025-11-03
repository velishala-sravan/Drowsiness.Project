#  Real-Time Driver Drowsiness Detection System

This is a real-time driver drowsiness detection system. It uses a webcam and a deep learning model to track the driver's eyes, and automatically plays an alarm to alert them if they start to fall asleep at the wheel.

The goal of this project is to prevent road accidents caused by driver fatigue.

Drivers falling asleep at the wheel is a major cause of accidents. This system acts as an "active safety" co-pilot that monitors the driver's state of alertness. By sounding an alarm before the driver completely falls asleep, it provides a critical warning that can prevent a serious or fatal crash.

##  Features

* **Real-Time Face & Eye Detection:** Uses Haar Cascade classifiers to find the driver's face and eyes from a webcam feed.
* **Deep Learning Model:** Employs a CNN trained to classify eyes as **'Open'** or **'Closed'**.
* **Drowsiness Score:** Maintains a "score" that increases when eyes are closed and decreases when they are open.
* **Audio & Visual Alarm:** Triggers a looping alarm sound (`alarm.wav`) and a flashing red border on the screen if the score exceeds a set threshold.
* **Tunable Sensitivity:** The drowsiness score threshold can be easily adjusted in the code to make the alarm more or less sensitive.

---

##  Technology Stack I Used:

* **Python 3.11+**
* **TensorFlow (Keras):** For building and training the CNN model.
* **OpenCV (cv2):** For real-time video capture, image processing, and detection.
* **Pygame:** For playing the alarm sound.
* **Numpy:** For numerical operations on image data.

---

##  Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

* [Git](https://git-scm.com/downloads)
* [Python 3.11](https://www.python.org/downloads/) (or 3.9-3.12)
* A webcam

### 2. Clone the Repository

Clone this repository to your local machine.
```bash
git clone [https://github.com/velishala-sravan/Drowsiness.Project.git](https://github.com/velishala-sravan/Drowsiness.Project.git) 
