# Import necessary libraries
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the saved model
model = load_model('uniform_detection_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Path to the specific test image
# test1
# image_path = r"C:\Users\rdpto\Desktop\uniform detection project\Data\uniform\IMG_20250214_131005.jpg"

# testing2

image_path = r"C:\Users\rdpto\Desktop\uniform detection project\Data\nonuniform\WhatsApp Image 2025-02-14 at 22.17.28_9244f03f.jpg"

# testing 3
# image_path =r"C:\Users\rdpto\Desktop\uniform detection project\Data\uniform\IMG_20250214_130940.jpg"

# Preprocess the image
processed_image = preprocess_image(image_path)

# Make a prediction
prediction = model.predict(processed_image)

# Print the result
if prediction > 0.5:
    print("Uniform Detected")
else:
    print("Non-Uniform Detected")