import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

# Set the paths to your training and test data folders
train_data_path = "train"
test_data_path = "test1"

# Preprocess the training data
train_data = []
for filename in os.listdir(train_data_path):
    img = cv2.imread(os.path.join(train_data_path, filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    train_data.append(img)
train_data = np.array(train_data).astype('float32') / 255.0

# Preprocess the test data
test_data = []
for filename in os.listdir(test_data_path):
    img = cv2.imread(os.path.join(test_data_path, filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    test_data.append(img)
test_data = np.array(test_data).astype('float32') / 255.0

np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)

# Split the training data into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Define the autoencoder model
inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(train_data, train_data,
                epochs=50,
                batch_size=128,
                shuffle=True,
                verbose=1,
                validation_data=(val_data, val_data))
autoencoder.save('autoencoder.h5')
# Evaluate the autoencoder on the test data
mse = autoencoder.evaluate(test_data, test_data)
print("Test MSE:", mse)

# Select a random sample of test data
sample_idx = random.sample(range(len(test_data)), 3)

# Generate and compare upscaled images
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

for i, idx in enumerate(sample_idx):
    img = test_data[idx]
    img = np.expand_dims(img, axis=0)
    upscaled = autoencoder.predict(img)
    upscaled = np.squeeze(upscaled, axis=0)
    true_upscaled = cv2.resize(img.squeeze(), (56, 56))

    # Display original, autoencoder-generated, and true upscaled images
    axs[i][0].imshow(img.squeeze(), cmap='gray')
    axs[i][0].set_title("Original")
    axs[i][1].imshow(upscaled, cmap='gray')
    axs[i][1].set_title("Autoencoder-generated")
    axs[i][2].imshow(true_upscaled, cmap='gray')
    axs[i][2].set_title("True upscaled")

plt.tight_layout()
plt.show()