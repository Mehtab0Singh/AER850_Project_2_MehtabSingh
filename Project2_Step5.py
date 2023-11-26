
# Step 5: Model Testing
from Project2_Steps1to4 import model
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2

model.summary()

# Load and preprocess the first image
test_medium = 'Project 2 Data/Data/Test/Medium/Crack__20180419_06_19_09,915.bmp'  # first image
medium_crack = image.load_img(test_medium, target_size=(100, 100))
medium_crack_array = image.img_to_array(medium_crack)
medium_crack_array = np.expand_dims(medium_crack_array, axis=0)
medium_crack_array /= 255.0  # Normalize pixel values to [0, 1]

# Load and preprocess the second image
test_large = 'Project 2 Data/Data/Test/Large/Crack__20180419_13_29_14,846.bmp'  # second image
large_crack = image.load_img(test_large, target_size=(100, 100))
large_crack_array = image.img_to_array(large_crack)
large_crack_array = np.expand_dims(large_crack_array, axis=0)
large_crack_array /= 255.0  # Normalize pixel values to [0, 1]

# Make predictions
mediumC_predictions = model.predict(medium_crack_array)[0]
largeC_predictions = model.predict(large_crack_array)[0]

# Define class labels
class_labels = ['Large Crack', 'Medium Crack', 'Small Crack', 'No Crack']


mediumC_top_labels = []
mediumC_top_probs = []
largeC_top_labels = []
largeC_top_probs = []

# Iterate through the predictions and save the top three
for i in range(3):
    top_idx = np.argmax(mediumC_predictions)
    mediumC_top_labels.append(class_labels[top_idx])
    mediumC_top_probs.append(mediumC_predictions[top_idx])
    mediumC_predictions[top_idx] = 0  # Set the top prediction to zero to find the next one

# Iterate through the predictions and save the top three
for i in range(3):
    top_idx = np.argmax(largeC_predictions)
    largeC_top_labels.append(class_labels[top_idx])
    largeC_top_probs.append(largeC_predictions[top_idx])
    largeC_predictions[top_idx] = 0  # Set the top prediction to zero to find the next one


# Ready predictions for the medium crack image
#for label, prob in zip(class_labels, mediumC_predictions):
 #   medium_crack_label = f"{label}: {prob * 100:.2f}%"

# Ready predictions for the large crack image
#for label, prob in zip(class_labels, largeC_predictions):
  #  large_crack_label = f"{label}: {prob * 100:.2f}%"

# Display the images with text
plt.figure(figsize=(100, 50))

plt.subplot(1, 2, 1)
plt.imshow((imageio.imread(test_medium)).astype(np.uint8))
#plt.title('Test Crack Image 1 with Predictions')
plt.text(100, 100, f"Medium Crack Test", color='white', fontsize=100)
plt.text(100, 200, f"1st Prediction: {mediumC_top_labels[0]}: {mediumC_top_probs[0] * 100:.2f}%", color='white', fontsize=100)
plt.text(100, 300, f"2nd Prediction: {mediumC_top_labels[1]}: {mediumC_top_probs[1] * 100:.2f}%", color='white', fontsize=100)
plt.text(100, 400, f"3rd Prediction: {mediumC_top_labels[2]}: {mediumC_top_probs[2] * 100:.2f}%", color='white', fontsize=100)
#plt.text(100, 100, f"1. {mediumC_top_labels[0]}: {mediumC_top_probs[0] * 100:.2f}%", color='white', fontsize=100, bbox=dict(facecolor='green', alpha=0.5))
plt.axis('on')

plt.subplot(1, 2, 2)
plt.imshow((imageio.imread(test_large)).astype(np.uint8))
#plt.title('Test Crack Image 2 with Predictions')
#plt.text(100, 100, large_crack_label, color='white', fontsize=100, bbox=dict(facecolor='green', alpha=0.5))
plt.text(100, 100, f"Large Crack Test", color='white', fontsize=100)
plt.text(100, 200, f"1st Prediction: {largeC_top_labels[0]}: {largeC_top_probs[0] * 100:.2f}%", color='white', fontsize=100)
plt.text(100, 300, f"2nd Prediction: {largeC_top_labels[1]}: {largeC_top_probs[1] * 100:.2f}%", color='white', fontsize=100)
plt.text(100, 400, f"3rd Prediction: {largeC_top_labels[2]}: {largeC_top_probs[2] * 100:.2f}%", color='white', fontsize=100)
plt.axis('on')

plt.show()
