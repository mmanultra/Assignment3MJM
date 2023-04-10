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

# randomly select one image from the test data
idx = np.random.randint(0, len(test_data))
image = test_data[idx]

# resize the image to 56x56 using cv2
true_56 = cv2.resize(image, (56, 56), interpolation=cv2.INTER_CUBIC)

# preprocess the image for the autoencoder
image_28 = np.expand_dims(image, axis=0)
image_28 = np.expand_dims(image_28, axis=-1)

# generate the autoencoder's 28x28 and 56x56 predictions
predicted_28 = autoencoder.predict(image_28)
predicted_56 = cv2.resize(predicted_28[0].squeeze(), (56, 56), interpolation=cv2.INTER_CUBIC)

# display the images side by side
fig, ax = plt.subplots(1,4, figsize=(12,3))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original 28x28')
ax[0].set_aspect('equal')

ax[1].imshow(true_56, cmap='gray')
ax[1].set_title('True 56x56')
ax[1].set_aspect('equal')

ax[2].imshow(predicted_28.squeeze(), cmap='gray')
ax[2].set_title('Autoencoder 28x28')
ax[2].set_aspect('equal')

ax[3].imshow(predicted_56, cmap='gray')
ax[3].set_title('Autoencoder 56x56')
ax[3].set_aspect('equal')

plt.tight_layout()
plt.savefig("my_plot_main.png")
plt.show()
