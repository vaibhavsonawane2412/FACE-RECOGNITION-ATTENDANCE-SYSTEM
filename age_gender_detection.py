import cv2
import numpy as np
import sqlite3
import os

def init_db():
    # Ensure the database directory exists
    db_dir = 'database'
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Connect to the SQLite database (or create it if it doesn't exist)
    db_path = os.path.join(db_dir, 'users.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Create the users table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT,
                name TEXT,
                email TEXT
            )
        ''')

        # Create the age_gender_detection table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS age_gender_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gender TEXT,
                age TEXT
            )
        ''')

        # Commit the changes
        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while initializing the database: {e}")
    finally:
        # Close the connection
        conn.close()

def save_to_database(results):
    conn = sqlite3.connect('database/users.db')
    cursor = conn.cursor()
    try:
        for result in results:
            cursor.execute('''
                INSERT INTO age_gender_detection (gender, age) VALUES (?, ?)
            ''', (result[0], result[1]))
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error occurred while inserting into database: {e}")
    finally:
        conn.close()

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

# Load models
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(3-7)', '(8-12)', '(13-18)', '(19-25)', '(26-35)', '(36-50)', '(51-65)', '(65+)']

genderList = ['Male', 'Female']

# Initialize the database
init_db()

# Use webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Using DirectShow backend 

# Check if the webcam opened successfully
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

padding = 20
results = []

while True:
    ret, frame = video.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read frame.")
        break

    frame, bboxs = faceBox(faceNet, frame)
    
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        # Ensure the bounding box is within the frame boundaries
        face = frame[max(0, y1-padding):min(y2+padding, frame.shape[0]-1), max(x1-padding, 0):min(x2+padding, frame.shape[1]-1)]
        
        # Check if face region is valid
        if face.shape[0] > 0 and face.shape[1] > 0:
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predict gender
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            # Predict age
            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            # Store results
            results.append((gender, age))

            # Display label and bounding box
            label = "{},{}".format(gender, age)
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame using OpenCV
    cv2.imshow('Age and Gender Detection', frame)

    # Save results to database
    save_to_database(results)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Print results
print("Detected age and gender:")
for result in results:
    print("Gender: {}, Age: {}".format(result[0], result[1]))
