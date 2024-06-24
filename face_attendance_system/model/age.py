import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split



# Define the path to the dataset
path = r"C:\Users\Vaibhav\face_attendance_system\datasets\UTKFace"  # Raw string prefix 'r' prevents escaping
images = []
ages = []
genders = []

# Load and preprocess the dataset
for img_name in os.listdir(path):
    age = int(img_name.split("_")[0])
    gender = int(img_name.split("_")[1])
    img = cv2.imread(os.path.join(path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))  # Resize image
    img = img / 255.0  # Scale image
    
    images.append(img)
    ages.append(age)
    genders.append(gender)
    
images = np.array(images, dtype=np.float32)
ages = np.array(ages, dtype=np.int64)
genders = np.array(genders, dtype=np.uint64)

# Group ages into bins
age_bins = [0, 18, 30, 45, 60, 100]
age_labels = [0, 1, 2, 3, 4]
ages_binned = np.digitize(ages, bins=age_bins, right=True)

# Shuffle and split the data into training and testing sets
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, ages_binned, test_size=0.2, random_state=42)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, genders, test_size=0.2, random_state=42, stratify=genders)

# Define and train the age model
age_model = Sequential([
    Conv2D(128, kernel_size=3, activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(256, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(512, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Flatten(),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dense(1, activation='linear', name='age')
])

age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
age_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_age = age_model.fit(x_train_age, y_train_age,
                            validation_data=(x_test_age, y_test_age),
                            epochs=35, callbacks=[early_stopping])

age_model.save('face_attendance_system\model_output\age_model.keras')

# Plot the training and validation loss for age model
plt.figure()
plt.plot(history_age.history['loss'], label='Training loss')
plt.plot(history_age.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss for Age Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation MAE for age model
plt.figure()
plt.plot(history_age.history['mae'], label='Training MAE')
plt.plot(history_age.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE for Age Model')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Define and train the gender model
gender_model = Sequential([
    Conv2D(36, kernel_size=3, activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(256, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(512, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Flatten(),
    Dropout(0.2),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid', name='gender')
])

gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gender_model.summary()

history_gender = gender_model.fit(x_train_gender, y_train_gender,
                                  validation_data=(x_test_gender, y_test_gender),
                                  epochs=35, callbacks=[early_stopping])

gender_model.save('face_attendance_system\model_output\gender_model.keras')

# Plot the training and validation accuracy for gender model
plt.figure()
plt.plot(history_gender.history['accuracy'], label='Training Accuracy')
plt.plot(history_gender.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy for Gender Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the gender model
gender_model = load_model('gender_model.keras', compile=False)
predictions = gender_model.predict(x_test_gender)
y_pred = (predictions >= 0.5).astype(int)[:, 0]

print("Accuracy =", accuracy_score(y_test_gender, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test_gender, y_pred)
sns.heatmap(cm, annot=True)
plt.show()
