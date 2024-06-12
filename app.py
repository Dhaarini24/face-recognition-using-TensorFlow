import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import get_file

# Download the LFW dataset
lfw_dataset_path = get_file(
    'lfw-deepfunneled.tgz',
    'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz',
    untar=True
)

# Load dataset
def load_lfw_dataset(dataset_path):
    images = []
    labels = []
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (160, 160))
                    images.append(image)
                    labels.append(person_name)
    return np.array(images), np.array(labels)

# Load the LFW dataset
dataset_path = os.path.join(os.path.dirname(lfw_dataset_path), 'lfw-deepfunneled')
images, labels = load_lfw_dataset(dataset_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Preprocess data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the model
model.save('face_recognition_model.h5')

# Function to recognize a face
def recognize_face(image_path, model, label_encoder):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (160, 160))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions)
    predicted_person = label_encoder.inverse_transform([predicted_label])
    
    return predicted_person[0]

# Load the saved model
model = tf.keras.models.load_model('face_recognition_model.h5')

# Recognize a face (example)
image_path = 'Example image for testing '
recognized_person = recognize_face(image_path, model, label_encoder)
print(f"Recognized Person: {recognized_person}")
