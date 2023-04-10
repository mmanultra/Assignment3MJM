import os
import numpy as np
from PIL import Image

# Define the paths to the training and testing folders
train_path = 'train'
test_path = 'test1'

# Define the desired dimensions for the images
train_dim = (28, 28)
test_dim = (56, 56)

# Load and process the images from the training folder
train_images = []
for filename in os.listdir(train_path):
    img = Image.open(os.path.join(train_path, filename))
    img = img.convert('L')  # Convert to black and white (grayscale)
    img = img.resize(train_dim)  # Resize the image
    train_images.append(np.array(img))

# Save the processed training images as numpy array
train_data = np.array(train_images)
np.save('train_data.npy', train_data)

# Load and process the images from the testing folder
test_images = []
for filename in os.listdir(test_path):
    img = Image.open(os.path.join(test_path, filename))
    img = img.convert('L')  # Convert to black and white (grayscale)
    img = img.resize(test_dim)  # Resize the image
    test_images.append(np.array(img))

# Save the processed testing images as numpy array
test_data = np.array(test_images)
np.save('test_data.npy', test_data)