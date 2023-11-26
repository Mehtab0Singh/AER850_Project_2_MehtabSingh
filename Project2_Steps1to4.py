# Import necessary modules and classes
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU

# Step 1: Data Processing
# Create a Sequential model
model = Sequential()

# Add an InputLayer with the desired input shape
model.add(InputLayer(input_shape=(100, 100, 3)))

# Add your layers as needed, e.g., Conv2D, Flatten, Dense, etc.
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))  # Output layer

# Compile the model and specify loss, optimizer, and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define data directories for training and validation data
train_data_dir = 'Project 2 Data/Data/Train'
validation_data_dir = 'Project 2 Data/Data/Validation'
test_data_dir = 'Project 2 Data/Data/Test'

# Define image target size
target_size = (100, 100)  # Set to (100, 100) from the project definition

# Define batch size
batch_size = 32

# Create an ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create train and validation data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)  # Only rescaling for validation data

# Add validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Create data augmentation for validation data using torchvision transforms
validation_transforms = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Add test data generator
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# validation data
validation_dataset = ImageFolder(root=validation_data_dir, transform=validation_transforms)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Step 2: Neural Network Architechture Design
# &
# Step 3: Hyperparameter Analysis
#model = models.Sequential()
#model.add(layers.Conv2D(100, (3, 3), activation='relu', input_shape=(100, 100, 3)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(200, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(200, (3, 3), activation='relu'))

# Define your model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(100, 100, 3)))
#model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU with a small negative slope
model.add(MaxPooling2D(pool_size=(2, 2)))  # MaxPooling2D layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # MaxPooling2D layer
model.add(Dropout(0.5))  # Adding dropout with a dropout rate of 0.5
model.add(Flatten())  # Flatten layer to connect to a dense layer
model.add(Dense(units=128, activation='relu'))
#model.add(LeakyReLU(alpha=0.2))  # Leaky ReLU with a small negative slope
model.add(Dropout(0.5))  # Adding dropout with a dropout rate of 0.5
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))  # Adding dropout with a dropout rate of 0.5
model.add(Dense(units=4, activation='softmax'))  # Considering the last dense layer must contain 4 neurons

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display the model summary
model.summary()

# Step 4: Model Evaluation

# Train the model
data_eval = model.fit(train_generator, epochs=20, validation_data=validation_generator)

# Evaluate the model on the test/validation data
eval_result = model.evaluate(test_generator)

# Display the evaluation result
print(f"Loss: {eval_result[0]}")
print(f"Accuracy: {eval_result[1]}")

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(data_eval.history['loss'])
plt.plot(data_eval.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(data_eval.history['accuracy'])
plt.plot(data_eval.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()
