
# .......................code 2.........................................

#
# # Import necessary libraries
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# import matplotlib.pyplot as plt
#
# # Verify GPU availability
# print("GPU available:", tf.config.list_physical_devices('GPU'))
#
# # Define paths
# train_dir = r"C:\Users\rdpto\Desktop\unifromData\train"  # Path to training dataset
# validation_dir = r"C:\Users\rdpto\Desktop\unifromData\validation"  # Path to validation dataset
#
# # Define constants
# IMAGE_SIZE = (128, 128)  # Resize images to 128x128
# BATCH_SIZE = 32  # Number of images processed in one batch
# EPOCHS = 10  # Number of training epochs
#
# # Data augmentation and preprocessing for training data
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
#     rotation_range=20,  # Randomly rotate images
#     width_shift_range=0.2,  # Randomly shift images horizontally
#     height_shift_range=0.2,  # Randomly shift images vertically
#     shear_range=0.2,  # Apply shear transformation
#     zoom_range=0.2,  # Randomly zoom images
#     horizontal_flip=True,  # Randomly flip images horizontally
#     fill_mode='nearest'  # Fill in missing pixels after transformations
# )
#
# # Preprocessing for validation data (only rescaling)
# validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
#
# # Load training data
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=IMAGE_SIZE,  # Resize images
#     batch_size=BATCH_SIZE,
#     class_mode='binary'  # Binary classification (uniform vs non-uniform)
# )
#
# # Load validation data
# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary'
# )
#
# # Build the CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # First convolutional layer
#     MaxPooling2D((2, 2)),  # First pooling layer
#     Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
#     MaxPooling2D((2, 2)),  # Second pooling layer
#     Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
#     MaxPooling2D((2, 2)),  # Third pooling layer
#     Flatten(),  # Flatten the output
#     Dense(128, activation='relu'),  # Fully connected layer
#     Dropout(0.5),  # Dropout to prevent overfitting
#     Dense(1, activation='sigmoid')  # Output layer (binary classification)
# ])
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // BATCH_SIZE,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // BATCH_SIZE,
#     epochs=EPOCHS
# )
#
# # Save the model
# model.save('uniform_detection_model.h5')
# print("Model saved as 'uniform_detection_model.h5'")
#
# # Plot training and validation accuracy/loss
# plt.figure(figsize=(12, 4))
#
# # Plot accuracy
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # Plot loss
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()





# ........................code 3...........................................................

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Verify GPU availability
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Define paths
train_dir = r"C:\Users\rdpto\Desktop\unifromData\train"  # Path to training dataset
validation_dir = r"C:\Users\rdpto\Desktop\unifromData\validation"  # Path to validation dataset

# Define constants
IMAGE_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 32  # Number of images processed in one batch
EPOCHS = 10  # Number of training epochs

# Data augmentation and preprocessing for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Apply shear transformation
    zoom_range=0.2,  # Randomly zoom images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in missing pixels after transformations
)

# Preprocessing for validation data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,  # Resize images
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Binary classification (uniform vs non-uniform)
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # First convolutional layer
    MaxPooling2D((2, 2)),  # First pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D((2, 2)),  # Second pooling layer
    Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
    MaxPooling2D((2, 2)),  # Third pooling layer
    Flatten(),  # Flatten the output
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# Save the model
# model.save('uniform_detection_model.h5')
model.save('uniform_detection_model.keras')
print("Model saved as 'uniform_detection_model.keras'")

# # Plot training and validation accuracy/loss
# plt.figure(figsize=(12, 4))
#
# # Plot accuracy
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.plot(history.history['val_acc'], label='Validation Accuracy')  # For older versions
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # Plot loss
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()

# ............2.............

# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate the model on the validation dataset
# val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
#
# # Print the validation accuracy
# print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")