import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Function to capture samples for new user registration
def capture_samples_page(username):
    # Initialize OpenCV's VideoCapture
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a directory for the user if it doesn't exist
    user_dir = os.path.join('samples', username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    sample_count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Save the detected face region as a sample image
            sample_count += 1
            sample_path = os.path.join(user_dir, f'sample_{sample_count}.jpg')
            cv2.imwrite(sample_path, frame[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Capture Samples', frame)
        cv2.waitKey(1)

        # Capture 100 samples
        if sample_count >= 100:
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Function to extract faces from images
def extract_faces(samples_dir):
    faces = []
    labels = []
    user_dirs = [d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d))]

    for user_dir in user_dirs:
        user_path = os.path.join(samples_dir, user_dir)
        print(f"Processing user directory: {user_path}")
        for image_name in os.listdir(user_path):
            image_path = os.path.join(user_path, image_name)
            print(f"Reading image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image file {image_path}. Skipping.")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(detected_faces) == 0:
                print(f"No faces detected in image: {image_path}")
            for (x, y, w, h) in detected_faces:
                face = gray[y:y + h, x:x + w]
                faces.append(face)
                labels.append(user_dir)

    print(f"Total faces extracted: {len(faces)}")
    return faces, labels



# Function to get face embeddings
def get_embeddings(faces):
    embeddings = []
    for face in faces:
        face_resized = cv2.resize(face, (160, 160))
        face_flattened = face_resized.flatten()
        embeddings.append(face_flattened)
    return np.array(embeddings)

# Function to train the face recognition model
def train_face_recognition_model():
    faces, labels = extract_faces('samples')
    embeddings = get_embeddings(faces)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels_encoded)
    joblib.dump(model, 'face_recognition_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

# Function to mark attendance
def mark_attendance(username):
    # Log attendance in the database
    log_attendance(username)

def log_attendance(username):
    db_path = os.path.join('database', 'attendance.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear all old records
    cursor.execute('DELETE FROM attendance')
    
    # Log the new attendance record
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       username TEXT NOT NULL,
                       timestamp TEXT NOT NULL)''')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('INSERT INTO attendance (username, timestamp) VALUES (?, ?)', (username, timestamp))
    conn.commit()
    conn.close()


# Function to detect emotion
def detect_emotion():
    # Detect emotion using a pre-trained model
    pass
