import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sqlite3

# Load the trained model
my_model = load_model('models/my_model_emotion.keras')

# Define the emotions
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Connect to the database
conn = sqlite3.connect('database/users.db')
cursor = conn.cursor()

while True:
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to match the input size of the model
    resized_frame = cv2.resize(gray, (48, 48))
    
    # Normalize the frame
    normalized_frame = resized_frame / 255.0
    
    # Reshape the frame to match the input shape of the model
    reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))
    
    # Predict emotion
    predictions = my_model.predict(reshaped_frame)
    emotion_label = class_labels[np.argmax(predictions)]
    
    # Display the emotion prediction on the frame
    cv2.putText(frame, emotion_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Save the predicted emotion to the database when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Predicted emotion:", emotion_label)
        
        # Insert the predicted emotion into the database
        try:
            cursor.execute('INSERT INTO emotion_detection (emotion) VALUES (?)', (emotion_label,))
            conn.commit()
            print("Emotion saved to database.")
        except sqlite3.Error as e:
            conn.rollback()
            print("Error saving emotion to database:", e)
        
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Close the database connection
conn.close()
