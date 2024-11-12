import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
from tensorflow.keras.models import Model
# Load and preprocess the data
base_dir = r"C:\Users\Rohit Sharma\Videos\archive\Training"
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 128

data = []
labels = []

# Load images and labels
for category in categories:
    path = os.path.join(base_dir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = load_img(os.path.join(path, img), target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img_array)
            data.append(img_array)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image: {e}")

# Convert lists to numpy arrays
data = np.array(data, dtype='float32') / 255.0  # Normalize data
labels = to_categorical(np.array(labels))  # One-hot encode labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Create a CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes for the 4 categories
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# Train the CNN model
history = cnn_model.fit(X_train, y_train, epochs=10, validation_split=0.2)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Extract features from the CNN model
feature_extractor = Sequential(cnn_model.layers[:-2])  # Remove the last two layers
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_features, np.argmax(y_train, axis=1))

# Evaluate the SVM model
y_pred = svm_model.predict(X_test_features)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
print("Accuracy:", accuracy_score(np.argmax(y_test, axis=1), y_pred))

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def preprocess_image(image_path, img_size=IMG_SIZE):
    """Preprocesses a single image for prediction."""
    img = load_img(image_path, target_size=(img_size, img_size))  # Load and resize the image
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_tumor_type(image_path):
    """Predicts the type of tumor in a given image."""
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Extract features using the CNN feature extractor
    features = feature_extractor.predict(img_array)
    features = features.reshape(1, -1)  # Flatten the features for SVM
    
    # Use the SVM model to predict the tumor type
    prediction = svm_model.predict(features)
    predicted_class = categories[prediction[0]]
    
    return predicted_class

import matplotlib.pyplot as plt

# Plot the training and validation accuracy
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,8))
for i in range(15):
  plt.subplot(3,5,i+1)
  plt.imshow(X_test[i],cmap='gray')
  plt.title(f"Predicted:{y_pred[i]}\nTrue:{y_test[i]}")
  plt.axis('off')
plt.show()

