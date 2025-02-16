from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('uniform_detection_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess the image and make a prediction
    processed_image = preprocess_image(file_path)
    prediction = model.predict(processed_image)

    # Determine the result
    if prediction > 0.5:
        result = "Uniform Detected"
    else:
        result = "Non-Uniform Detected"

    # Return the result as JSON
    return jsonify({'result': result})

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)