import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import dlib

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load dlib's face detector and shape predictor
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

# Function to extract faces from images
def extract_faces(samples_dir):
    images = []
    labels = []

    for person_dir in os.listdir(samples_dir):
        if os.path.isdir(os.path.join(samples_dir, person_dir)):
            for filename in os.listdir(os.path.join(samples_dir, person_dir)):
                image_path = os.path.join(samples_dir, person_dir, filename)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    aligned_face = align_face(image, (x, y, w, h))
                    images.append(cv2.resize(aligned_face, (100, 100)))  # Resize for consistency
                    labels.append(person_dir)
    
    return images, labels

# Function to align face
def align_face(image, rect):
    (x, y, w, h) = rect
    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
    landmarks = shape_predictor(image, dlib_rect)
    aligned_face = dlib.get_face_chip(image, landmarks, size=100)
    return aligned_face

# Function to train the face recognition model
def train_model(samples_dir):
    images, labels = extract_faces(samples_dir)
    if not images:
        print("No faces found in the sample directory.")
        return
    
    # Convert labels to integers
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    # Normalize the images
    images_normalized = [img / 255.0 for img in images]

    # Train SVM classifier
    svm = SVC(kernel='linear', probability=True)
    svm.fit(np.array(images_normalized).reshape(len(images_normalized), -1), labels_encoded)

    # Save the trained model and label encoder
    joblib.dump(svm, 'models/face_recognition_model.pkl')
    joblib.dump(encoder, 'models/label_encoder.pkl')

# Load the trained model and label encoder
def load_model():
    svm = joblib.load('models/face_recognition_model.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    return svm, encoder

# Function to recognize faces in an image
def recognize_face(image_path):
    svm, encoder = load_model()
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    recognized = False
    predicted_label = None

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        aligned_face = align_face(image, (x, y, w, h))
        face_resized = cv2.resize(aligned_face, (100, 100))  # Resize for consistency
        face_normalized = face_resized / 255.0  # Normalize the face
        label = svm.predict_proba(face_normalized.reshape(1, -1))
        predicted_label = encoder.inverse_transform(label.argmax(axis=1))[0]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        recognized = True
        break  # Assuming only one face per image

    # Save the result image
    result_image_path = os.path.join('recognized_faces', 'result.jpg')
    if not os.path.exists('recognized_faces'):
        os.makedirs('recognized_faces')
    cv2.imwrite(result_image_path, image)

    if recognized:
        return predicted_label, result_image_path
    else:
        return None, None

if __name__ == "__main__":
    # Train the model
    train_model('samples')
