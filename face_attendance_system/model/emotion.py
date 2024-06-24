import matplotlib
matplotlib.use('Agg')  # Use 'Agg' for environments without a display (e.g., servers)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import tensorflow as tf

IMG_HEIGHT = 48
IMG_WIDTH = 48
batch_size = 32

train_data_dir = r"C:\Users\Vaibhav\face_attendance_system\datasets\data\train"

validation_data_dir = r"C:\Users\Vaibhav\face_attendance_system\datasets\data\test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# Wrap the generators in tf.data.Dataset and use repeat
train_dataset = tf.data.Dataset.from_generator(lambda: train_generator,
                                               output_signature=(
                                                   tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
                                                   tf.TensorSpec(shape=(None, 7), dtype=tf.float32)
                                               )).repeat()

validation_dataset = tf.data.Dataset.from_generator(lambda: validation_generator,
                                                    output_signature=(
                                                        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
                                                        tf.TensorSpec(shape=(None, 7), dtype=tf.float32)
                                                    )).repeat()

# Verify our generator by plotting a few faces and printing corresponding labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img, label = next(iter(train_dataset))

i = random.randint(0, (img.shape[0])-1)
image = img[i]
labl = class_labels[np.argmax(label[i])]
plt.imshow(image[:, :, 0], cmap='gray')
plt.title(labl)
plt.savefig('sample_image.png')  # Save the sample image plot

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

num_train_imgs = sum([len(files) for _, _, files in os.walk(train_data_dir)])
num_test_imgs = sum([len(files) for _, _, files in os.walk(validation_data_dir)])

epochs = 50

history = model.fit(
    train_dataset,
    steps_per_epoch=num_train_imgs // batch_size,
    epochs=epochs,
    validation_data=validation_dataset,
    validation_steps=num_test_imgs // batch_size
)

model.save('my_model_emotion.keras')

# Plot the training and validation accuracy and loss at each epoch
def plot_history(history, metric, validation_metric, metric_name, file_name):
    plt.figure()
    plt.plot(history[metric], 'y', label=f'Training {metric_name}')
    plt.plot(history[validation_metric], 'r', label=f'Validation {metric_name}')
    plt.title(f'Training and validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

plot_history(history.history, 'loss', 'val_loss', 'loss', 'loss_plot.png')
plot_history(history.history, 'accuracy', 'val_accuracy', 'accuracy', 'accuracy_plot.png')

# Test the model
from tensorflow.keras.models import load_model

# Correctly load the saved model
my_model = load_model('my_model_emotion.keras')

# Generate a batch of images
test_img, test_lbl = next(iter(validation_dataset))
predictions = my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

# Calculate accuracy
print("Accuracy = ", accuracy_score(test_labels, predictions))

# Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Check results on a few select images
n = random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_labl = class_labels[test_labels[n]]
pred_labl = class_labels[predictions[n]]
plt.imshow(image[:, :, 0], cmap='gray')
plt.title(f"Original label: {orig_labl} - Predicted: {pred_labl}")
plt.savefig('test_sample_result.png')
plt.close()
