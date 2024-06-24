import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import dlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten

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

# Corrected data augmentation function
def augment_images(images, labels):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    augmented_images, augmented_labels = [], []
    for img, label in zip(images, labels):
        img = img.reshape((1,) + img.shape + (1,)) if img.ndim == 2 else img.reshape((1,) + img.shape)
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            i += 1
            if i >= 5:  # Generate 5 augmented images per input image
                break
    return np.array(augmented_images), np.array(augmented_labels)

# Function to train the face recognition model using transfer learning
def train_model(samples_dir):
    images, labels = extract_faces(samples_dir)
    if not images:
        print("No faces found in the sample directory.")
        return
    
    # Augment the dataset
    augmented_images, augmented_labels = augment_images(images, labels)
    
    # Convert labels to integers
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(augmented_labels)

    # Normalize and preprocess the images
    images_normalized = np.array([preprocess_input(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB)) for img in augmented_images])

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images_normalized, labels_encoded, test_size=0.2, random_state=42)

    # Load VGG16 model pre-trained on ImageNet and exclude the top fully connected layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

    # Add custom top layers
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(len(set(labels_encoded)), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    # Save the trained model
    model.save('models/face_recognition_model.keras')
    joblib.dump(encoder, 'models/label_encoder.pkl')

# Load the trained model and label encoder
def load_model_and_encoder():
    model = load_model('models/face_recognition_model.keras')
    encoder = joblib.load('models/label_encoder.pkl')
    return model, encoder

# Function to recognize faces in an image using the trained model
def recognize_face(image_path):
    model, encoder = load_model_and_encoder()
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    recognized = False
    predicted_label = None

    for (x, y, w, h) in faces:
        face = image_rgb[y:y+h, x:x+w]
        aligned_face = align_face(image_rgb, (x, y, w, h))
        face_resized = cv2.resize(aligned_face, (100, 100))  # Resize for consistency
        face_preprocessed = preprocess_input(face_resized)  # Preprocess for VGG16
        face_preprocessed = np.expand_dims(face_preprocessed, axis=0)  # Add batch dimension
        prediction = model.predict(face_preprocessed)
        predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]
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
