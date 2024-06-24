import cv2
import mediapipe as mp
import sqlite3
import os
from datetime import datetime

# Mediapipe and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Database setup
db_dir = 'database'
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

db_path = os.path.join(db_dir, 'users.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the landmarks table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS landmarks_detection (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        jawline_detected INTEGER,
        eyebrows_detected INTEGER,
        nose_detected INTEGER,
        eyes_detected INTEGER,
        lips_detected INTEGER
    )
''')
conn.commit()

# Webcam capture setup
cap = cv2.VideoCapture(0)

def detect_landmarks(landmarks):
    jawline = landmarks[0:17]
    left_eyebrow = landmarks[17:22]
    right_eyebrow = landmarks[22:27]
    nose = landmarks[27:36]
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    outer_lips = landmarks[48:60]
    inner_lips = landmarks[60:68]

    return {
        "jawline_detected": int(len(jawline) == 17),
        "eyebrows_detected": int(len(left_eyebrow) == 5 and len(right_eyebrow) == 5),
        "nose_detected": int(len(nose) == 9),
        "eyes_detected": int(len(left_eye) == 6 and len(right_eye) == 6),
        "lips_detected": int(len(outer_lips) == 12 and len(inner_lips) == 8)
    }

with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
                
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
                
                landmarks = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in face_landmarks.landmark]
                detected_features = detect_landmarks(landmarks)

                # Insert data into the database
                cursor.execute('''
                    INSERT INTO landmarks_detection (jawline_detected, eyebrows_detected, nose_detected, eyes_detected, lips_detected) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (detected_features["jawline_detected"], detected_features["eyebrows_detected"], detected_features["nose_detected"], detected_features["eyes_detected"], detected_features["lips_detected"]))
                conn.commit()
        
        cv2.imshow("Facial Landmarks Detection", cv2.flip(image, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
